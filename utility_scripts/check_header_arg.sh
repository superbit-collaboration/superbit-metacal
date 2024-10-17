export hdr_key='IMG_QUAL'
export hdr_key2='LOCKED'

for file in *.sub.fits; do
    fitsheader $file | grep "$hdr_key" | awk -v fname="$file" '{print fname "\t" $0}' >> header_IMG.txt
done

for file in *.sub.fits; do
    fitsheader $file | grep "$hdr_key2" | awk -v fname="$file" '{print fname "\t" $0}' >> header_LOCKED.txt
done
