pip3 install fairseq sacremoses fastBPE subword_nmt textdistance[extras] scipy tqdm beautifulsoup4 sacrebleu
wget -c http://statmt.org/wmt14/test-full.tgz -O - | tar -xz
mv test-full/newstest2014-fren-src.en.sgm .
mv test-full/newstest2014-fren-ref.fr.sgm .
rm -rf test-full/
curl -O https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip -o multinli_1.0.zip
rm -rf multinli_1.0.zip __MACOSX/
curl -O https://www.unicode.org/Public/security/latest/intentional.txt