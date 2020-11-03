#!/usr/bin/env bash


shuf tweets_neg.txt | split -l $[ $(wc -l tweets_neg.txt | cut -d" " -f1) * 80 / 100 ]
mv xaa split/tweets_neg_train.txt
mv xab split/tweets_neg_test.txt

shuf tweets_pos.txt | split -l $[ $(wc -l tweets_pos.txt | cut -d" " -f1) * 80 / 100 ]
mv xaa split/tweets_pos_train.txt
mv xab split/tweets_pos_test.txt

wc -l *.txt

wc -l split/*.txt