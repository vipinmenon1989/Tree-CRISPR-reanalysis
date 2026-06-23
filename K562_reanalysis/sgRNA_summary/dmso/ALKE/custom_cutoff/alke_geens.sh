awk -F, '
BEGIN {
    print "=== GENES WITH SIGMOID SCORE > 0.25 ==="
}
NR > 1 {
    # If score is greater than 0.25
    if ($3 > 0.25) {
        if (!seen_gt[$2]++) {
            print $2 " (Score: " $3 ")"
            count_gt++
        }
    } 
    # If score is less than or equal to 0.25
    else {
        if (!seen_lt[$2]++) {
            print $2 " (Score: " $3 ") [LOW]"
            count_lt++
        }
    }
}
END {
    print "\n=== SUMMARY ==="
    printf "Unique Genes > 0.25: %d\n", count_gt + 0
    printf "Unique Genes <= 0.25: %d\n", count_lt + 0
}' ALKE_hits_NTC_new_cutoff.csv
