import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from preprocessing import *
from bound import BoundingBox


if __name__ == '__main__':
    img_path = 'samples/star.png'
    crop_path = 'cropped/cropstaff_0.png'

    solid_note_paths = ['templates/solid-note-1.png',
                        'templates/solid-note-1.png']

    half_note_paths = ['templates/half-note-1.png',
                       'templates/half-note-2.png',
                       'templates/half-note-3.png',
                       'templates/half-note-4.png']

    whole_note_paths = ['templates/whole-note-1.png',
                        'templates/whole-note-2.png',
                        'templates/whole-note-3.png',
                        'templates/whole-note-4.png']

    solid_note_templates = [cv2.imread(template_path, 0) for template_path in solid_note_paths]
    half_note_templates = [cv2.imread(template_path, 0) for template_path in half_note_paths]
    whole_note_templates = [cv2.imread(template_path, 0) for template_path in whole_note_paths]

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit('Error opening image!')

    # if img.shape[0] > 1400 or img.shape[1] > 1400:
    #      img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

    staffs = get_staffs(img, verbose=True)
    staves = remove_staves(img, staffs, verbose=False)
    lines = detect_lines(img, staves, staffs, verbose=True)

    solid_notes_positions = detect(img, staffs, solid_note_templates, 0.71, verbose=True)
    half_notes_positions = detect(img, staffs, half_note_templates, 0.71, verbose=True)
    whole_notes_positions = detect(img, staffs, whole_note_templates, 0.71, verbose=True)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
