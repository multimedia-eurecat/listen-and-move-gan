
suffix=checkpoints/0050000_Gn.model
checks_seyeon=(\
    ../20230105081409-study-seyeon-128-route_rnn-motion_L1basic-vp_none/$suffix \
    ../20230208152757-study-seyeon-128-route_rnngen-motion_L3v2basic-vp_dir-cond_gen/$suffix \
)

checks_onceIsaw=(\
    ../20230104160343-study-onceIsaw-128-route_rnn-motion_L1basic-vp_none/$suffix \
    ../20230128003550-study-onceIsaw-128-route_rnngen-motion_L3v2basic-vp_dir-cond_gen/$suffix \
)

check_m31=(\
    ../20230102175235-study-mead_m31-128-route_rnn-motion_L1basic-vp_none/$suffix \
    ../20230113072746-study-mead_M31-128-route_rnngen-motion_L3basic-vp_dir-cond_gen/$suffix \
)

check_drums=(\
    ../20230206212610-study-drums-128-route_rnn-motion_L1v2basic-vp_none/$suffix \
    ../20230208151320-study-drums-128-route_rnngen-motion_L3v2basic-vp_dir-cond_gen/$suffix \
)

assets=(
    ../assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
    # ../assets/MEAD_W33_left_60-CUT.mov \
    ../assets/MEAD_M31_right_60.mp4 \
    ../assets/Once_I_Saw_CUT.mp4 \
    ../assets/Istanbul_Agop_16_Xist_Ion_Crash_Cymbal-Brilliant.mp4 \
)

checkpoints=("checks_seyeon" "check_m31" "checks_onceIsaw" "checks_drums")

crops=(
    650,400,1250,1000 \
    # 330,0,1410,1080 \
    500,0,1580,1080 \
    290,0,1010,720 \
    170,0,890,720 \
)

image_size=128
chunk_len=0.085
fps=20
seq_len=32

sound_route=(rnn rnngen)
motion_layers=(1 3)
motion_type=(basic basic)
vid_pred=("" dir)
cond_gen=("" --cond_gen)

for (( c=0; c<${#checkpoints[@]}; c++ ));
do
    asset=${assets[$c]}
    crop=${crops[$c]}
    declare -n checks="${checkpoints[c]}"
    echo "Source: "$asset

    for (( j=0; j<${#checks[@]}; j++ ));
    do
        echo "Checkpoint "${checks[$j]}
        check=${checks[$j]}
        route=${sound_route[$j]}
        mlayers=${motion_layers[$j]}
        mtype=${motion_type[$j]}
        vpred=${vid_pred[$j]}
        cgen=${cond_gen[$j]}

        python eval_FSD50K.py \
            --gpus 0 \
            --train_source $asset \
            --checkpoint $check \
            --crop $crop \
            --image_size $image_size \
            --chunk_len $chunk_len \
            --fps $fps \
            --seq_len $seq_len \
            --sound_route $route \
            --motion_layers $mlayers \
            --motion_type $mtype \
            --vid_pred $vpred \
            $cgen
    done
done

# python eval_FSD50K.py \
#     --gpu 3 \
#     --train_video ../assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
#     --checkpoint ../20220924155610-seyeon-basic128-mel64-ninject/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 650,400,1250,1000

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
#     --checkpoint ../20220905150945-seyeon-basic128-mel64/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 650,400,1250,1000

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
#     --checkpoint ../20220905151013-seyeon-basic128-mel128/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 128 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 650,400,1250,1000

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/seyeon_jung-JSBach_cello_suite_No6_CUT.mp4 \
#     --checkpoint ../20220905151024-seyeon-basic128-mfcc/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mfcc \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 650,400,1250,1000

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/Obama-Weekly_Address_Ensuring_Our_Free_Market_Works_for_Everyone_CUT.mp4 \
#     --checkpoint ../20220905150602-obama-basic128-mel64/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 430,0,1510,1080

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/Obama-Weekly_Address_Ensuring_Our_Free_Market_Works_for_Everyone_CUT.mp4 \
#     --checkpoint ../20220905150922-obama-basic128-mel128/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 128 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 430,0,1510,1080

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/Obama-Weekly_Address_Ensuring_Our_Free_Market_Works_for_Everyone_CUT.mp4 \
#     --checkpoint ../20220905150942-obama-basic128-mfcc/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mfcc \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 430,0,1510,1080

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/JACOB_COLLIER_improvises_in_Db_MAJOR_PAD.mp4 \
#     --checkpoint ../20220905151037-jacob-basic128-mel64/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 0,0,1280,1280

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/JACOB_COLLIER_improvises_in_Db_MAJOR_PAD.mp4 \
#     --checkpoint ../20220905151112-jacob-basic128-mel128/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 128 \
#     --feat_type mel \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 0,0,1280,1280

# python eval_FSD50K.py \
#     --gpu 1 \
#     --train_video ../assets/JACOB_COLLIER_improvises_in_Db_MAJOR_PAD.mp4 \
#     --checkpoint ../20220905151125-jacob-basic128-mfcc/checkpoints/0200000_Gn.model \
#     --chunk_len 0.085 \
#     --mel_bands 64 \
#     --feat_type mfcc \
#     --e_motion 10 \
#     --image_size 128 \
#     --g_arch basic \
#     --crop 0,0,1280,1280
