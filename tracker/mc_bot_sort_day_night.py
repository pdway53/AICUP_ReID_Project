import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface
from yolov7.utils.plots import plot_one_box
import math


#判斷兩項量是否有相交匯
def Intersect(l1, l2):
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    a = v0[0] * v1[1] - v0[1] * v1[0]
    b = v0[0] * v2[1] - v0[1] * v2[0]

    temp = l1
    l1 = l2
    l2 = temp
    v1 = (l1[0] - l2[0], l1[1] - l2[1])
    v2 = (l1[0] - l2[2], l1[1] - l2[3])
    v0 = (l1[0] - l1[2], l1[1] - l1[3])
    c = v0[0] * v1[1] - v0[1] * v1[0]
    d = v0[0] * v2[1] - v0[1] * v2[0]

    if a*b < 0 and c*d < 0:
        return True
    else:
        return False
    


#add matching penalty to unreasonable matching
def check_dist2(dists,strack_pool ,detections,strack_pool_1,detections_1,matches):

    matches_line = []

    for itracked, idet in matches:
        track = strack_pool_1[itracked]
        det = detections_1[idet]
        det_center_x = det._tlwh[0] + det._tlwh[2]/2
        det_center_y = det._tlwh[1] + det._tlwh[3]/2
        track_center_x = track._tlwh[0] + track._tlwh[2]/2
        track_center_y = track._tlwh[1] + track._tlwh[3]/2

        motion_x = det_center_x - track_center_x
        motion_y = det_center_y - track_center_y

        matches_line.append([track_center_x,track_center_y,det_center_x,det_center_y])

    for j in range(len(dists)):    
        for i in range(len(dists[0])):
            track = strack_pool[j]
            det = detections[i]            
            det_center_x = det._tlwh[0] + det._tlwh[2]/2
            det_center_y = det._tlwh[1] + det._tlwh[3]/2
            track_center_x = track._tlwh[0] + track._tlwh[2]/2
            track_center_y = track._tlwh[1] + track._tlwh[3]/2
            motion_x = det_center_x - track_center_x
            motion_y = det_center_y - track_center_y
            estimate_x = int(track_center_x - track.start_pos_x)
            estimate_y = int(track_center_y - track.start_pos_y)
            theta = cos([estimate_x,estimate_y],[motion_x,motion_y])
            direct_angle = dot_product_angle( np.array([estimate_x,estimate_y]),np.array([motion_x,motion_y] ))             
            if direct_angle>75 and track.is_activated and (np.abs(motion_x) > track._tlwh[2]*0.25 or np.abs(motion_y) > track._tlwh[3]*0.25) :                
                dists[j][i] = 1

            #if second match line cross first association match line
            line2 = [track_center_x,track_center_y,det_center_x,det_center_y]
            for line in matches_line:
                if Intersect(line,line2)==True:
                    dists[j][i] += 0.25    # 

    return dists

def check_out_boudary(lost_track):
    
    mv_x = lost_track.cur_position[0] - lost_track.start_pos_x
    mv_y = lost_track.cur_position[0] - lost_track.start_pos_x
    # x, y, w, h, vx, vy, vw, vh
    if lost_track.mean[4]*mv_x > 0 and lost_track.mean[5]*mv_y > 0 and lost_track.is_activated:
        if (lost_track.mean[0] < 1280*0.05 or lost_track.mean[0] > 1280*0.9) and mv_x > 1280*0.3:
            return True
        if lost_track.mean[1] < 720*0.05 or lost_track.mean[1] > 720*0.9 and mv_x > 720*0.3:
            return True

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.cur_position = (tlwh[0] + tlwh[2]/2 ,tlwh[1] + tlwh[3]/2)
        self.last_motion_x = None
        self.last_motion_y = None
        self.activate_times = 0
        self.start_pos_x = None
        self.start_pos_y = None
        self.last_match_frameID = None
        self.match_score = 0
        self.is_static = False
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_features_adjust(self, feat,alpha):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.activate_times = self.activate_times+1
        self.start_pos_x = self._tlwh[0] + self._tlwh[2]/2
        self.start_pos_y = self._tlwh[1] + self._tlwh[3]/2

        self.last_match_frameID = frame_id
        self.last_motion_x = 0
        self.last_motion_y = 0

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features_adjust(new_track.curr_feat, 0.8)  #0.8

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.activate_times = self.activate_times+1
        self.last_match_frameID = frame_id

        self.last_motion_x = new_track._tlwh[0] + new_track._tlwh[2]/2 - self._tlwh[0] - self._tlwh[2]/2
        self.last_motion_y = new_track._tlwh[0] + new_track._tlwh[2]/2 - self._tlwh[0] - self._tlwh[2]/2
        self._tlwh = new_track._tlwh

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id, score = 0):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.last_motion_x = new_track._tlwh[0] + new_track._tlwh[2]/2 - self._tlwh[0] - self._tlwh[2]/2
        self.last_motion_y = new_track._tlwh[0] + new_track._tlwh[2]/2 - self._tlwh[0] - self._tlwh[2]/2
         
        new_tlwh = new_track.tlwh
        old_tlwh = self._tlwh
        #add if static
        scale_ratio_x = (float)(new_track._tlwh[2])/(float)(self._tlwh[2])
        if (abs(self.last_motion_x) <5 and abs(self.last_motion_y) <5 and abs(new_track._tlwh[2] - self._tlwh[2]) < 5 and abs(new_track._tlwh[3] - self._tlwh[3]) <5   ):
            self.is_static =True
        else:
            self.is_static =False
        
        self.last_match_frameID = frame_id
        # update new bbox axis from yolo
        self._tlwh = new_tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        #if self.mean[]
        if self.last_motion_y < 15 and  self.last_motion_x <15 :
            alpha = 0.9  #alpha = 0.9
        elif self.last_motion_y > 50 or self.last_motion_x >100 or self.last_motion_y > old_tlwh[2]*3 or self.last_motion_y > old_tlwh[2]*3 or new_tlwh[2]/old_tlwh[2] > 4 or new_tlwh[2]/old_tlwh[2] > 3 :
            alpha = 0.65 #alpha = 0.70.8
        else:
            alpha = 0.75 #

        

        if new_track.curr_feat is not None:
            self.update_features_adjust(new_track.curr_feat, alpha)
            #self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)
        self.activate_times = self.activate_times+1
        self.cur_position = (new_track._tlwh[0] + new_track._tlwh[2]/2 ,new_track._tlwh[1] + new_track._tlwh[3]/2) 
        self.match_score = score


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        # if self.mean is None:
        #     return self._tlwh.copy()
        # ret = self.mean[:4].copy()
        # ret[:2] -= ret[2:] / 2
        # return ret

        # We disable the kalman filter output
        return self._tlwh.copy()

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def cos(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
    return sum([array1[i]*array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)

def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        #print("Zero magnitude vector!")
        return 0
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0



class BoTSORT_ADJUST(object):
    def __init__(self, args, frame_rate=30,prev_cnt=0):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count(prev_cnt)

        self.frame_id = 0
        self.args = args
        self.id_cnt = 0

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh
        self.match_thresh = self.args.match_thresh
        self.match_thresh2    = 0.7
        self.appearance_thresh = args.appearance_thresh
        self.lost_latency =2
        self.prev_all_stracks = []

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = 240 #self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results, img, frameID=0,day = 1,args=None,is_change_camera=0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if day:
            self.track_high_thresh = args.track_high_thresh
            self.track_low_thresh = args.track_low_thresh
            self.new_track_thresh = args.new_track_thresh
            self.match_thresh = self.args.match_thresh
            self.match_thresh2    = 0.7
            self.appearance_thresh = args.appearance_thresh
            self.lost_latency = 3
        else:
            self.track_high_thresh = args.track_high_thresh_night
            self.track_low_thresh = args.track_low_thresh_night
            self.new_track_thresh = args.new_track_thresh_night
            self.match_thresh = self.args.match_thresh_night
            self.match_thresh2   = 0.85
            self.appearance_thresh = 0.7 #args.appearance_thresh
            self.lost_latency = 4

        if len(output_results):
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            features = output_results[:, 6:]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = output_results[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]

        if is_change_camera == 0:
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)
        else:
            self.lost_stracks = []
            self.tracked_stracks = []


        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0

            # original
            # emb_dists[ious_dists_mask] = 1.0
            # dists = np.minimum(ious_dists, emb_dists)

            # Only use reid feature as distance(without bbox iou)
            dists = emb_dists

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        
        for j in range(len(dists)):    
            for i in range(len(dists[0])):
                track = strack_pool[j]
                det = detections[i]               
                det_center_x = det._tlwh[0] + det._tlwh[2]/2
                det_center_y = det._tlwh[1] + det._tlwh[3]/2
                track_center_x = track._tlwh[0] + track._tlwh[2]/2
                track_center_y = track._tlwh[1] + track._tlwh[3]/2
                motion_x = det_center_x - track_center_x
                motion_y = det_center_y - track_center_y
                estimate_x = int(track_center_x - track.start_pos_x)
                estimate_y = int(track_center_y - track.start_pos_y)
                scale_ratio_w = det._tlwh[2]/track._tlwh[2]
                scale_ratio_h = det._tlwh[3]/track._tlwh[3]                
                direct_angle = dot_product_angle( np.array([estimate_x,estimate_y]),np.array([motion_x,motion_y] ))

                #use angle constrain penalty : add penalty to the match pair which make car direction unreasonable
                if direct_angle>75 and track.is_activated and (np.abs(motion_x) > track._tlwh[2]*0.25 or np.abs(motion_y) > track._tlwh[3]*0.25) :
                    dists[j][i] += 1
                

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        strack_pool_1 =strack_pool
        detections_1 = detections
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]

            det_center_x = det._tlwh[0] + det._tlwh[2]/2
            det_center_y = det._tlwh[1] + det._tlwh[3]/2
            track_center_x = track._tlwh[0] + track._tlwh[2]/2
            track_center_y = track._tlwh[1] + track._tlwh[3]/2

            motion_x = det_center_x - track_center_x
            motion_y = det_center_y - track_center_y     

            estimate_x = int(track_center_x - track.start_pos_x)
            estimate_y = int(track_center_y - track.start_pos_y)          

           
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id,dists[itracked][idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]

            if self.args.with_reid:
                features_second = self.encoder.inference(img, dets_second)

        else:
            dets_second = []
            scores_second = []
            classes_second = []
            features_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''            
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                                 (tlbr, s, c, f) in zip(dets_second, scores_second, classes_second, features_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # bbox iou distance
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # Used ReID features
        #dists = matching.embedding_distance(r_tracked_stracks, detections_second) / 2.0
        #dists[dists > self.appearance_thresh] = 1.0   


        #Second association
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        unconfirmed.extend(r_tracked_stracks)
        detections = [detections[i] for i in u_detection]
        dists = matching.embedding_distance(unconfirmed, detections) 
        dists2 = check_dist2(dists,unconfirmed,detections,strack_pool_1,detections_1 ,matches)
  
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        
       
        matches2, u_unconfirmed, u_detection = matching.linear_assignment(dists2, thresh=self.match_thresh2) #0.7  0.5
      
        for itracked, idet in matches2:
            track = unconfirmed[itracked]            
            unconfirmed[itracked].update(detections[idet], self.frame_id,dists2[itracked][idet])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]

            track.mark_lost()
            lost_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            #跨相機re matching
            #prev_trakcer = [detections[i] for i in self.prev_all_stracks]
            #dists3 = matching.embedding_distance(detections[inew], prev_trakcer) 
            #matches3, u_unconfirmed3, u_detection3 = matching.linear_assignment(dists2, thresh=0.2)            

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        self.lost_stracks = sub_stracks(self.lost_stracks,refind_stracks)
        for track in self.lost_stracks:

            if track.is_static == True and (self.frame_id -track.last_match_frameID <= self.lost_latency):
                continue 
            if self.frame_id - track.last_match_frameID >  self.lost_latency:           
                track.mark_removed()
                removed_stracks.append(track)
                continue               
            
            #remove 跑出邊框的框條件
            track_center_x = track._tlwh[0] + track._tlwh[2]/2
            track_center_y = track._tlwh[1] + track._tlwh[3]/2
            
            est_vx = track.mean[4]
            est_vy = track.mean[5]
            if track.is_activated and ( track_center_x + track.last_motion_x < 1280*0.05 or track_center_y + track.last_motion_y < 720*0.05 or  track_center_x + track.last_motion_x > 1280*0.95 or track_center_y + track.last_motion_y > 720*0.95): 
                track.mark_removed()
                removed_stracks.append(track)
                continue
            
            if check_out_boudary(track):
                track.mark_removed()
                removed_stracks.append(track)
                continue
            

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.tracked_stracks = sub_stracks(self.tracked_stracks, self.removed_stracks)
        self.tracked_stracks = sub_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        self.prev_all_stracks.extend(self.removed_stracks)
        self.id_cnt = BaseTrack._count
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
