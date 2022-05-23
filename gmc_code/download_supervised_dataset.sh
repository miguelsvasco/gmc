# MOSEI dataset

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ssmt7pe9BA6P-ilB148BqtC89d9d9rK0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ssmt7pe9BA6P-ilB148BqtC89d9d9rK0" -O ./supervised/dataset/mosei_train_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1U-MfdQRFO2GBIANyuqmyYF_mHE-OCl5z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1U-MfdQRFO2GBIANyuqmyYF_mHE-OCl5z" -O ./supervised/dataset/mosei_valid_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zyJpvwAw2RyZG54eZ0I8Bzs0sR_zsdgs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zyJpvwAw2RyZG54eZ0I8Bzs0sR_zsdgs" -O ./supervised/dataset/mosei_test_a.dt && rm -rf /tmp/cookies.txt


# MOSI dataset

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RdHPXQ7XOx7fj3vO1kVL4yqFch5srYK7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RdHPXQ7XOx7fj3vO1kVL4yqFch5srYK7" -O ./supervised/dataset/mosi_train_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QwDrzycIY6EPeCVcEvpsFULy2Dg8etYG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QwDrzycIY6EPeCVcEvpsFULy2Dg8etYG" -O ./supervised/dataset/mosi_valid_a.dt && rm -rf /tmp/cookies.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14nT4DHrbPzknq35TMMiYjgWO7Go-ubj-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14nT4DHrbPzknq35TMMiYjgWO7Go-ubj-" -O ./supervised/dataset/mosi_test_a.dt && rm -rf /tmp/cookies.txt
