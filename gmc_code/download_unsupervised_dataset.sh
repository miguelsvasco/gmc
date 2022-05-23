# Multimodal Handwritten Digits dataset

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV" -O ./unsupervised/dataset/mhd_train.pt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6" -O ./unsupervised/dataset/mhd_test.pt && rm -rf /tmp/cookies.txt
