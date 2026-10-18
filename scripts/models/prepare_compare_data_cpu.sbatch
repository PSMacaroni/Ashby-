#!/bin/bash
# Get every dataset the backbone comparison needs, on a CPU node. (Delta's login nodes cut the session when
# one user does heavy file work there, so downloads and big copies belong in a job like this.)
#
#   - STARC-9: the four sets in parallel (resumes: tiles already on disk are skipped), then one more pass
#     to fill in any tile that failed
#   - CRC-VAL-HE-7K + Kather 2016 from Zenodo (skipped if already there)
#   - SPIDER: if SPIDER_FROM is set, copy (or check) a teammate's extracted tiles, file by file, re-copying
#     anything missing or a different size. Same tiles scripts/models/get_spider.py picks (same seed).
#
# From the project folder on Delta:
#     mkdir -p data/logs
#     sbatch --export=ALL,SPIDER_FROM=/projects/bfqi/<teammate>/datasets scripts/models/prepare_compare_data_cpu.sbatch
#SBATCH --job-name=prepare_compare_data
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
export LC_ALL=C          # same sort order everywhere (the file lists are compared with comm)
D=data/datasets
L=data/logs
J=$SLURM_JOB_ID

echo "== $(date +%T) STARC-9 (4 sets in parallel) + Zenodo"
for s in train val_stanford_small val_stanford_large val_tcga; do
    python -W ignore scripts/models/get_starc9.py --out $D/starc9 --sets $s --workers 24 > $L/get_starc9_$s-$J.log 2>&1 &
done
python -W ignore scripts/models/get_public_tests.py --out $D > $L/get_public_tests-$J.log 2>&1 &

if [ -n "$SPIDER_FROM" ]; then
    for o in spider_colorectal spider_skin; do
        mkdir -p $D/$o && chmod 700 $D/$o              # gated dataset (CC-BY-NC): keep it private
        (cd $SPIDER_FROM/$o && find . -type f -printf "%P %s\n" | sort) > $L/spider_src_$o-$J.txt
        (cd $D/$o && find . -type f -printf "%P %s\n" | sort) > $L/spider_dst_$o-$J.txt
        comm -23 $L/spider_src_$o-$J.txt $L/spider_dst_$o-$J.txt | cut -d" " -f1 > $L/spider_todo_$o-$J.txt
        echo "SPIDER $o: $(wc -l < $L/spider_src_$o-$J.txt) files at the source, $(wc -l < $L/spider_todo_$o-$J.txt) missing or different here"
        TODO=$PWD/$L/spider_todo_$o-$J.txt
        DST=$PWD/$D/$o
        (cd $SPIDER_FROM/$o && xargs -a "$TODO" -r -P 12 -n 100 cp -f --parents -t "$DST/")
    done
fi
wait
echo "== $(date +%T) second STARC-9 pass (fills any failed tile)"
python -W ignore scripts/models/get_starc9.py --out $D/starc9 --workers 24 > $L/get_starc9_pass2-$J.log 2>&1
tail -c 2000 $L/get_starc9_pass2-$J.log | tr "\r" "\n" | grep -E "done|failed" | tail -8

echo "== $(date +%T) counts"
for s in train val_stanford_small val_stanford_large val_tcga; do
    echo "STARC-9 $s: $(find $D/starc9/$s -name '*.png' | wc -l) tiles"
done
echo "CRC-VAL-HE-7K: $(find $D/CRC-VAL-HE-7K -type f | wc -l) files"
echo "Kather 2016: $(find $D/kather2016 -type f | wc -l) files"
ok=1
if [ -n "$SPIDER_FROM" ]; then
    for o in spider_colorectal spider_skin; do
        (cd $D/$o && find . -type f -printf "%P %s\n" | sort) > $L/spider_dst_$o-$J.txt
        n=$(comm -23 $L/spider_src_$o-$J.txt $L/spider_dst_$o-$J.txt | wc -l)
        echo "SPIDER $o: $(wc -l < $L/spider_dst_$o-$J.txt) files, $n still missing or different"
        [ "$n" -eq 0 ] || ok=0
    done
fi
[ -d $D/CRC-VAL-HE-7K ] && [ -d $D/kather2016 ] || ok=0
[ $ok -eq 1 ] && echo "DATA READY" || { echo "DATA NOT READY"; exit 1; }
