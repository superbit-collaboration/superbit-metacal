export hdr_key='STREHL'
export hdr_key2='PSF_FWHM'

for file in *.fits; do
    fitsheader $file | grep "$hdr_key" | awk -v fname="$file" '{print fname "\t" $0}' >> header_val.txt
done

for file in *.fits; do
    fitsheader $file | grep "$hdr_key2" | awk -v fname="$file" '{print fname "\t" $0}' >> header_val.txt
done
