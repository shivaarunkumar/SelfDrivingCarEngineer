(track1_cimages,track1_limages,track1__rimages,track1_sangles)=ImportImageData('..\\..\\Track1\\New\\Center\\driving_log.csv')
(track1_s2s_cimages,track1_s2s_limages,track1__s2s_rimages,track1_s2s_sangles)=ImportImageData('..\\..\\Track1\\New\\Swerve\\driving_log.csv')
(track1_rev_cimages,track1_rev_limages,track1__rev_rimages,track1_rev_sangles)=ImportImageData('..\\..\\Track1\\New\\Reverse\\driving_log.csv')
(track1_cur_cimages,track1_cur_limages,track1__cur_rimages,track1_cur_sangles)=ImportImageData('..\\..\\Track1\\New\\Curves\\driving_log.csv')

augmented_images,augmented_measurements = [], []
correction = 0.2

for image,limage,rimage,sangle in zip(track1_cimages,track1_limages,track1__rimages,track1_sangles):
    augmented_images.append(image)
    augmented_measurements.append(sangle)
    # Add flipped
    flipped_image = cv2.flip(image,1) # Flip Horizontal
    augmented_images.append(flipped_image)
    augmented_measurements.append(sangle*-1.0)
    # Add left
    augmented_images.append(limage)
    augmented_measurements.append(sangle+correction)
    # Add Right
    augmented_images.append(rimage)
    augmented_measurements.append(sangle-correction)


# for image,limage,rimage,sangle in zip(track1_s2s_cimages,track1_s2s_limages,track1__s2s_rimages,track1_s2s_sangles):
#     augmented_images.append(image)
#     augmented_measurements.append(sangle)
#     # Add flipped
#     flipped_image = cv2.flip(image,1) # Flip Horizontal
#     augmented_images.append(flipped_image)
#     augmented_measurements.append(sangle*-1.0)
#     # Add left
#     augmented_images.append(limage)
#     augmented_measurements.append(sangle+correction)
#     # Add Right
#     augmented_images.append(rimage)
#     augmented_measurements.append(sangle-correction)

# for image,limage,rimage,sangle in zip(track2_cimages,track2_limages,track2__rimages,track2_sangles):
#     augmented_images.append(image)
#     augmented_measurements.append(sangle)    
    # # Add left
    # augmented_images.append(limage)
    # augmented_measurements.append(sangle+correction)
    # # Add Right
    # augmented_images.append(rimage)
    # augmented_measurements.append(sangle-correction)