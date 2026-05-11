awk '
    # When the file changes, print the name and record count of the PREVIOUS file
    FNR == 1 && NR > 1 { 
        print prev_file, prev_fnr 
    } 
    # Update variables for every line
    { 
        prev_file = FILENAME
        prev_fnr = FNR 
    } 
    # Print the very last file at the end
    END { 
        print prev_file, prev_fnr 
    }' *
