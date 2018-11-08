import os
import cv2
import logging
import motmetrics as mm
from tracker.mot_tracker import OnlineTracker

from datasets.mot_seq import get_loader
from utils import visualization as vis
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(dataloader, data_type, result_filename, save_dir=None, show_image=True):
    if save_dir is not None:
        mkdirs(save_dir)

    tracker = OnlineTracker()
    timer = Timer()
    results = []
    wait_time = 1
    for frame_id, batch in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        frame, det_tlwhs, det_scores, _, _ = batch

        # run tracking
        timer.tic()
        online_targets = tracker.update(frame, det_tlwhs, None)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)

        key = cv2.waitKey(wait_time)
        key = chr(key % 128).lower()
        if key == 'q':
            exit(0)
        elif key == 'p':
            cv2.waitKey(0)
        elif key == 'a':
            wait_time = int(not wait_time)

    # save results
    write_results(result_filename, results, data_type)


def main(data_root='/data/MOT16/train', det_root=None,
         seqs=('MOT16-05',), exp_name='demo', save_image=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdirs(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None

        logger.info('start seq: {}'.format(seq))
        loader = get_loader(data_root, det_root, seq)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        eval_seq(loader, data_type, result_filename,
                 save_dir=output_dir, show_image=show_image)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    metrics = mm.metrics.motchallenge_metrics
    # metrics = None
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, f'summary_{exp_name}.xlsx'))

    # # eval
    # try:
    #     import matlab.engine as matlab_engine
    #     eval_root = '/data/MOT17/amilan-motchallenge-devkit'
    #     seqmap = 'eval_mot_generated.txt'
    #     with open(os.path.join(eval_root, 'seqmaps', seqmap), 'w') as f:
    #         f.write('name\n')
    #         for seq in seqs:
    #             f.write('{}\n'.format(seq))
    #
    #     logger.info('start eval {} in matlab...'.format(result_root))
    #     eng = matlab_engine.start_matlab()
    #     eng.cd(eval_root)
    #     eng.run_eval(data_root, result_root, seqmap, '', nargout=0)
    # except ImportError:
    #     logger.warning('import matlab.engine failed...')


if __name__ == '__main__':
    # import fire
    # fire.Fire(main)

    seqs_str = '''MOT16-02
                MOT16-05
                MOT16-09
                MOT16-11
                MOT16-13'''
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(data_root='/data/MOT16/train',
         seqs=seqs,
         exp_name='mot16_val',
         show_image=False)
