for p in $(seq 0.1 0.2 1); do 
    p_formatted=$(printf "%.3f" $p)
    python lp_agcn_tri.py --pr 0.7 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --predp 0.05 --gnndp 0.05  --probscale 4.3 --proboffset 2.8 --alpha 1.0  --gnnlr 0.0043 --prelr 0.0024  --batch_size 1152  --ln --lnnn --predictor cn1.1 --model puregcn --hiddim 256 --mplayers 1  --testbs 8192  --maskinput  --jk  --use_xlin  --tailact --epochs 9999 --runs 2
done