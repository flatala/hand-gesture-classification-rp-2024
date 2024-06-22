import os
import xml.etree.ElementTree as ET
import random
import sys
import cv2
import shutil
from collections import defaultdict

# This script works for the following folder structure:
#
#                    main_folder|
#                               |sub_folder_1|
#                                            | annotations_for_segment.xml
#                                            | video_segment.mp4
#
#                               |sub_folder_2|
#                                            | annotations_for_segment.xml
#                                            | video_segment.mp4
#                                             
# Provide target main_folder path + output_folder path when running from terminal. 
# This should extract the video segments corresponding to the annotations, shuffle them and split into train, test and validation sets
#
# Important: GPT-4o model was used to write some parts of this code + for debugging purposes

def run(root_fir, dest_dir):

    processed_lh_dir = os.path.join(dest_dir, 'Processed_LH')
    processed_rh_dir = os.path.join(dest_dir, 'Processed_RH')
    os.makedirs(processed_lh_dir, exist_ok=True)
    os.makedirs(processed_rh_dir, exist_ok=True)

    for subdir, dirs, files in os.walk(root_fir):
        ann_file = None
        vid_file = None

        for name in os.listdir(subdir):
            if name.startswith('annotations') and name.endswith('.xml'):
                ann_file = os.path.join(subdir, name)
            elif name.startswith('cam') and name.endswith(('.mp4')):
                vid_file = os.path.join(subdir, name)

        if ann_file and vid_file:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            seg_lh = {}
            tr_lh = root.find(f'.//TIER[@columns="TrajDirection_LH"]')

            if tr_lh is not None:
                for span in tr_lh.findall('span'):
                    start = float(span.get('start'))
                    end = float(span.get('end'))
                    lbl = span.find('v').text
                    if lbl not in seg_lh:
                        seg_lh[lbl] = []
                    seg_lh[lbl].append((start, end))

            seg_rh = {}
            tr_rh = root.find(f'.//TIER[@columns="TrajDirection_RH"]')
            if tr_rh is not None:
                for span in tr_rh.findall('span'):
                    start = float(span.get('start'))
                    end = float(span.get('end'))
                    lbl = span.find('v').text
                    if lbl not in seg_rh:
                        seg_rh[lbl] = []
                    seg_rh[lbl].append((start, end))

            for seg, hand_dir in zip([seg_lh, seg_rh], [processed_lh_dir, processed_rh_dir]):
                for label in seg:
                    os.makedirs(os.path.join(hand_dir, label), exist_ok=True)

            cap = cv2.VideoCapture(vid_file)
            fps = cap.get(cv2.CAP_PROP_FPS)

            for seg, hand_dir in zip([seg_lh, seg_rh], [processed_lh_dir, processed_rh_dir]):
                for label in seg:
                    for i, (start, end) in enumerate(seg[label]):
                        rand = random.randint(0, 10000000)
                        new_path = os.path.join(hand_dir, label, f'{rand}_{label}_segment_{i}.mp4')

                        frame_s = int(start * fps)
                        frame_e = int(end * fps)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_s)
                        ret_frames = []

                        for _ in range(frame_s, frame_e + 1):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            ret_frames.append(frame)

                        if not ret_frames:
                            continue

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        h, w, _ = ret_frames[0].shape
                        out = cv2.VideoWriter(new_path, fourcc, fps, (w, h))

                        for frame in ret_frames:
                            out.write(frame)

                        out.release()

            cap.release()

    for hand in ['LH', 'RH']:
        base_dir = os.path.join(dest_dir, f'Processed_{hand}')
        lbls = os.listdir(base_dir)
        lbls = [label for label in lbls if os.path.isdir(os.path.join(base_dir, label))]
        all_vids = []
        for lbl in lbls:
            label_dir = os.path.join(base_dir, lbl)
            vids = os.listdir(label_dir)
            vids = [os.path.join(lbl, vid) for vid in vids]
            all_vids.extend(vids)

        random.shuffle(all_vids)
        train_split = int(0.7 * len(all_vids))
        val_split = int(0.85 * len(all_vids))
        train_vids = all_vids[:train_split]
        val_vids = all_vids[train_split:val_split]
        test_vids = all_vids[val_split:]

        def split_dirs(base, name, labels):
            split_dir = os.path.join(base, name)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            for label in labels:
                label_dir = os.path.join(split_dir, label)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
            return split_dir

        train_dir = split_dirs(os.path.join(dest_dir, hand), "train", lbls)
        val_dir = split_dirs(os.path.join(dest_dir, hand), "validation", lbls)
        test_dir = split_dirs(os.path.join(dest_dir, hand), "test", lbls)
        counters = defaultdict(int)

        for vid in train_vids:
            lbl, file_name = os.path.split(vid)
            src_file = os.path.join(base_dir, lbl, file_name)
            counters[lbl] += 1
            rand = random.randint(0, 10000000)
            ret_file = os.path.join(train_dir, lbl, f'{rand}_{lbl}_{counters[lbl]}.mp4')
            shutil.move(src_file, ret_file)
        for vid in val_vids:
            lbl, file_name = os.path.split(vid)
            src_file = os.path.join(base_dir, lbl, file_name)
            counters[lbl] += 1
            rand = random.randint(0, 10000000)
            ret_file = os.path.join(val_dir, lbl, f'{rand}_{lbl}_{counters[lbl]}.mp4')
            shutil.move(src_file, ret_file)
        for vid in test_vids:
            lbl, file_name = os.path.split(vid)
            src_file = os.path.join(base_dir, lbl, file_name)
            counters[lbl] += 1
            rand = random.randint(0, 10000000)
            ret_file = os.path.join(test_dir, lbl, f'{rand}_{lbl}_{counters[lbl]}.mp4')
            shutil.move(src_file, ret_file)


if __name__ == "__main__":
    dir = sys.argv[1]
    dest = sys.argv[2]
    run(dir, dest)
