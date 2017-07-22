export PYTHONPATH="`pwd`/../"
CONFIG="configs/skip/example.py" # TODO
TASK=$1

case $TASK in
  "train")
    GPUID=$2
    python run.py -c $CONFIG --task "$TASK" -g $GPUID
    ;;
  "eval")
    RESTORE_PATH=$2
    GPUID=$3
    python run.py -c $CONFIG --task "$TASK" --restore $RESTORE_PATH -g $GPUID
    ;;
  "continue")
    RESTORE_PATH=$2
    GPUID=$3
    python run.py -c $CONFIG --task "$TASK" --restore $RESTORE_PATH -g $GPUID
    ;;
  "eval_all_steps")
    MODEL_ID=$2
    GPUID=$3
    python run.py -c $CONFIG --task "$TASK" --model_id $MODEL_ID -g $GPUID
    ;;
  "list_result")
    MODEL_ID=$2
    python run.py -c $CONFIG --task "$TASK" --model_id $MODEL_ID
    ;;
  "svm")
    FEAT_PATH=$2
    # TODO
    python classifier/svm.py MED/MED14-MEDTEST-EK100.pkl $FEAT_PATH
    ;;
  "show_best")
    MODEL_ID=$2
    python run.py -c $CONFIG --task "$TASK" --model_id $MODEL_ID
    ;;
  "do_cls")
    RESTORE_PATH=$2
    GPUID=$3
    python run.py -c $CONFIG --task "$TASK" --restore $RESTORE_PATH -g $GPUID
    ;;
  *)
    echo "unknown task $TASK"
esac
