#!/bin/bash
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13qK2itKGjiUg_sTBWLKBdnUK2kxy2HC2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13qK2itKGjiUg_sTBWLKBdnUK2kxy2HC2" -O senteval_data.tar.gz && rm -rf ./cookies.txt
