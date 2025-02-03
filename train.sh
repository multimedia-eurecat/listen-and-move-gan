
# This script automates the training process for the AVGAN model.
# It iterates over a predefined set of assets, names, and crops,
# and invokes the training function with the appropriate parameters.
# Usage:
#   ./train.sh <number_of_gpus> <train_source> <trial_name> <sound_route> <motion_layers> <motion_type> <image_size> <video_prediction> <crop> <total_iter>
# Example:
#   ./train.sh 1 ./assets/myvideo.mp4 myname rnn 1 basic 128 "" 650,400,1250,1000 50000

train()
{
    VP=$([ -z "$8" ] && echo "-vp_none" || echo "-vp_$8")
    CG=$(echo ${11} | sed 's/-//')
    python train.py \
        --gpus $1 \
        --train_source $2 \
        --trial_name study-$3-$7-route_$4-motion_L$5$6$VP$CG \
        --sound_route $4 \
        --motion_layers $5 \
        --motion_type $6 \
        --image_size $7 \
        --vid_pred $8 \
        --crop $9 \
        --total_iter ${10} \
        ${11}
}

assets=(
    ./assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
    ./assets/Once_I_Saw_CUT.mp4 \
    ./assets/Istanbul_Agop_16_Xist_Ion_Crash_Cymbal-Brilliant.mp4 \
    ./assets/MEAD_W33_left_60-CUT.mov \
    ./assets/MEAD_M31_right_60.mp4 \
)
    
names=(\
    seyeon \
    onceIsaw \
    drums \
    mead_W33 \
    mead_M31 \
)

crops=(
    650,400,1250,1000 \
    290,0,1010,720 \
    30,0,750,720 \
    330,0,1410,1080 \
    500,0,1580,1080 \
)

iters=50000

for (( j=0; j<${#assets[@]}; j++ ));
do
    asset=${assets[$j]}
    name=${names[$j]}
    crop=${crops[$j]}

    if [ "$1" -eq "0" ]; then
        gpu=0
        train $gpu $asset $name rnn    1 basic  128 ""      $crop $iters     # seyeon
        train $gpu $asset $name rnn    3 basic  128 ""      $crop $iters
    fi
    if [ "$1" -eq "1" ]; then
        gpu=0
        train $gpu $asset $name rnn    1 acc    128 ""      $crop $iters
        train $gpu $asset $name rnngen 1 basic  128 ""      $crop $iters
    fi
    if [ "$1" -eq "2" ]; then
        gpu=1
        train $gpu $asset $name rnn    1 basic  128 ""      $crop $iters --cond_gen
        train $gpu $asset $name rnn    1 basic  128 basic   $crop $iters
    fi
    if [ "$1" -eq "3" ]; then
        gpu=1
        train $gpu $asset $name rnn    1 basic  128 dir     $crop $iters
    fi
done