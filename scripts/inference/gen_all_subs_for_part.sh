
for f in ../predictions/by_fold/*$1*.pkl
do
    result=`basename $f`
    result=${result%.pkl}.csv

    if [ ! -f $result ]
    then
        echo generation submission for $f
        ./gen_sub.py $f &
    fi
done
