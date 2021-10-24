pip3 install fairseq sacremoses fastBPE subword_nmt textdistance[extras] scipy tqdm beautifulsoup4 sacrebleu pandas google-api-python-client pyarrow
wget -c http://statmt.org/wmt14/test-full.tgz -O - | tar -xz
mv test-full/newstest2014-fren-src.en.sgm .
mv test-full/newstest2014-fren-ref.fr.sgm .
rm -rf test-full/
curl -O https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip
unzip -o multinli_1.0.zip
rm -rf multinli_1.0.zip __MACOSX/
curl -O https://www.unicode.org/Public/security/latest/intentional.txt
rm -rf assets/
rm -rf toxic/
mkdir assets
wget https://max-cdn.cdn.appdomain.cloud/max-toxic-comment-classifier/1.0.0/assets.tar.gz --output-document=assets/assets.tar.gz
tar -x -C assets/ -f assets/assets.tar.gz -v
rm assets/assets.tar.gz
git clone https://github.com/IBM/MAX-Toxic-Comment-Classifier.git
mv MAX-Toxic-Comment-Classifier toxic
sed -i 's/==.*//g' toxic/requirements.txt
pip install -r toxic/requirements.txt
pip install maxfw
sed -i 's/from config/from ..config/g' toxic/core/model.py
sed -i 's/from core\./from ./g' toxic/core/model.py
wget https://ndownloader.figshare.com/files/7394542 -O toxicity_annotated_comments.tsv
wget https://ndownloader.figshare.com/files/7394539 -O toxicity_annotations.tsv
pip3 install transformers datasets