## MOTDT

### Reference

```
@inproceedings{long2018tracking,
  title={Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification},
  author={Long, Chen and Haizhou, Ai and Zijie, Zhuang and Chong, Shang},
  year={2018},
  booktitle={ICME}
}
```

### Usage

Download MOT16 dataset and trained weights from the following links.
Put weight files in `data`, then build and run the code. 

```bash
pip install -r requirements.txt
sh make.sh
python eval_mot.py
```

I used five of six training sequences as the validation set.
Following are the details and evaluation results.  Please note that the results may be a little different with the paper because this is a re-implementation version.

```
Sequences: 
    'MOT16-02'
    'MOT16-05'
    'MOT16-09'
    'MOT16-11'
    'MOT16-13'

    ... MOT16-02
Preprocessing (cleaning) MOT16-02...
......
Removing 656 boxes from solution...
MOT16-02
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 38.0 76.4 25.3| 30.6  92.5  0.73|   54   7   20   27|   441 12379    47   146|  27.8  75.1  28.1 

    ... MOT16-05
Preprocessing (cleaning) MOT16-05...
........
Removing 1 boxes from solution...
MOT16-05
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 52.0 80.8 38.3| 44.3  93.3  0.26|  125  12   68   45|   216  3801    35   130|  40.6  76.1  41.1 

    ... MOT16-09
Preprocessing (cleaning) MOT16-09...
.....
Removing 765 boxes from solution...
MOT16-09
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 58.6 73.1 48.9| 63.2  94.5  0.37|   25   7   16    2|   195  1936    35    66|  58.8  75.2  59.4 

    ... MOT16-11
Preprocessing (cleaning) MOT16-11...
.........
Removing 2 boxes from solution...
MOT16-11
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 54.3 71.6 43.7| 57.7  94.5  0.34|   69  11   29   29|   309  3884    29    74|  54.0  79.3  54.3 

    ... MOT16-13
Preprocessing (cleaning) MOT16-13...
.......
Removing 0 boxes from solution...
MOT16-13
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 38.0 71.7 25.9| 29.5  81.8  1.01|  107  11   39   57|   754  8072    46   178|  22.5  72.6  22.9 


 ********************* Your MOT16 Results *********************
 IDF1  IDP  IDR| Rcll  Prcn   FAR|   GT  MT   PT   ML|    FP    FN   IDs    FM|  MOTA  MOTP MOTAL 
 45.7 74.4 33.0| 40.5  91.4  0.53|  380  48  172  160|  1915 30072   192   594|  36.3  75.9  36.7
```

### Evaluate
You can use official [matlab eval devkit](https://bitbucket.org/amilan/motchallenge-devkit/) to evaluate the outputs.
Or directly use the python version [motmetrics](https://github.com/cheind/py-motmetrics).
I already added the python evaluation method in the `eval_mot.py` script.
The results are slightly different from the official devkit since the ignoring method is not identical.
Results from python evaluation:
```
          IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs   FM  MOTA  MOTP
MOT16-02 37.1% 75.6% 24.6% 30.2% 93.0%  54  7  21  26  406 12440  47  146 27.7% 0.247
MOT16-05 53.7% 83.0% 39.7% 44.6% 93.1% 125 13  68  44  224  3779  35  130 40.8% 0.242
MOT16-09 61.1% 75.8% 51.1% 63.6% 94.3%  25  8  15   2  202  1913  28   64 59.2% 0.247
MOT16-11 54.9% 72.2% 44.3% 58.1% 94.7%  69 12  28  29  301  3840  27   70 54.6% 0.208
MOT16-13 38.2% 71.6% 26.0% 29.7% 81.6% 107 11  38  58  766  8051  46  178 22.6% 0.276
OVERALL  46.1% 75.1% 33.3% 40.6% 91.5% 380 51 170 159 1899 30023 183  588 36.5% 0.241
```

### Resources

Paper: Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-identification ([researchgate](https://www.researchgate.net/publication/326224594_Real-time_Multiple_People_Tracking_with_Deeply_Learned_Candidate_Selection_and_Person_Re-identification), [arxiv](https://arxiv.org/abs/1809.04427))

Results on the test set: https://motchallenge.net/tracker/MOTDT

Eval Devkit: https://bitbucket.org/amilan/motchallenge-devkit/

Models: https://drive.google.com/open?id=1ETfqSoy7OeT-8GO75F1bYWhP3mzrwwvn
