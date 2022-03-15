echo 'div2k-x2' &&
python test.py --config ./configs/test/test-fast-div2k-2.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x3' &&
python test.py --config ./configs/test/test-fast-div2k-3.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x4' &&
python test.py --config ./configs/test/test-fast-div2k-4.yaml --fast True --model $1 --gpu $2 &&

echo 'div2k-x6*' &&
python test.py --config ./configs/test/test-fast-div2k-6.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x12*' &&
python test.py --config ./configs/test/test-fast-div2k-12.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x18*' &&
python test.py --config ./configs/test/test-fast-div2k-18.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x24*' &&
python test.py --config ./configs/test/test-fast-div2k-24.yaml --fast True --model $1 --gpu $2 &&
echo 'div2k-x30*' &&
python test.py --config ./configs/test/test-fast-div2k-30.yaml --fast True --model $1 --gpu $2 &&

true
