import cv2
import numpy as np
import sys
import argparse
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt
from preprocessing import *
from bound import BoundingBox

import warnings
warnings.filterwarnings("ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sheet", required=True,
        help="path to input image")
    ap.add_argument("-n", "--name", type=str, default="melody",
        help="name of the song")
    ap.add_argument("-v", "--verbose", type=bool, default=False,
        help="show detailed report")
    args = vars(ap.parse_args())

    img_path = args["sheet"]
    enable_verbose = args["verbose"]
    name = args["name"]

    sharp_paths = ['templates/sharp-1.png',
                   'templates/sharp-2.png']

    flat_paths = ['templates/flat-1.png']

    solid_note_paths = ['templates/solid-note-1.png',
                        'templates/solid-note-2.png',
                        'templates/solid-note-3.png']

    half_note_paths = ['templates/half-note-1.png',
                       'templates/half-note-2.png',
                       'templates/half-note-3.png',
                       'templates/half-note-4.png']

    whole_note_paths = ['templates/whole-note-1.png',
                        'templates/whole-note-2.png',
                        'templates/whole-note-3.png',
                        'templates/whole-note-4.png']

    sharp_templates = [cv2.imread(template_path, 0) for template_path in sharp_paths]
    flat_templates = [cv2.imread(template_path, 0) for template_path in flat_paths]
    solid_note_templates = [cv2.imread(template_path, 0) for template_path in solid_note_paths]
    half_note_templates = [cv2.imread(template_path, 0) for template_path in half_note_paths]
    whole_note_templates = [cv2.imread(template_path, 0) for template_path in whole_note_paths]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        sys.exit('Error opening image!')

    if img.shape[0] > 1500 or img.shape[1] > 1500:
        img = cv2.resize(img, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

    print('[INFO] Staffs searching...')
    staffs = get_staffs(img, verbose=enable_verbose)
    # thresh = search_staffs(img, staffs, verbose=True)
    print('[INFO] Getting staves...')
    staves = get_staves(img, staffs, verbose=enable_verbose)

    print('[INFO] Getting lines positions')
    lines_positions = detect_lines(img, staves, staffs, verbose=enable_verbose)

    print('[INFO] Accidentals detection...')
    sharp_positions = detect(img, staffs, sharp_templates, 0.71, verbose=enable_verbose)
    # flat_positions = detect(img, staffs, flat_templates, 0.90, verbose=False)

    print('[INFO] Solid notes detection...')
    solid_notes_positions = detect(img, staffs, solid_note_templates, 0.71, verbose=enable_verbose)

    print('[INFO] Half notes detection...')
    half_notes_positions = detect(img, staffs, half_note_templates, 0.71, verbose=enable_verbose)

    print('[INFO] Whole notes detection...')
    whole_notes_positions = detect(img, staffs, whole_note_templates, 0.71, verbose=enable_verbose)

    print('[INFO] Merging...')
    sharp_positions = [merge_boxes(bounding_boxes, 0.5) for bounding_boxes in sharp_positions]
    # flat_positions = [merge_boxes(bounding_boxes, 0.5) for bounding_boxes in flat_positions]
    solid_notes_positions = [merge_boxes(bounding_boxes, 0.5) for bounding_boxes in solid_notes_positions]
    half_notes_positions = [merge_boxes(bounding_boxes, 0.5) for bounding_boxes in half_notes_positions]
    whole_notes_positions = [merge_boxes(bounding_boxes, 0.5) for bounding_boxes in whole_notes_positions]

    img_copy = np.copy(img)
    for boxes in solid_notes_positions:
        for box in boxes:
            box.draw(img_copy, (0, 200, 0), 2)
    cv2.imwrite("detection.png", img_copy)

    print('[INFO] Durations definition...')
    durations = define_durations(img, staffs, solid_notes_positions, verbose=enable_verbose)
    print('[INFO] Pitch definition...')

    all_notes = []
    # flat_notes = [Note(Note.NOTES[1][0], Note.NOTES[1][1], None, None)]
    flat_notes = []
    # sharp_notes = []
    for i, lines in enumerate(lines_positions):
        sharp_notes = get_pitches(staffs[i], lines, sharp_positions[i])
        # sharp_notes = []
        solid_notes = get_pitches(staffs[i], lines, solid_notes_positions[i], sharp_notes, flat_notes, duration=durations[i])
        half_notes = get_pitches(staffs[i], lines, half_notes_positions[i], sharp_notes, flat_notes, duration=2)
        whole_notes = get_pitches(staffs[i], lines, whole_notes_positions[i], sharp_notes, flat_notes, duration=4)
        staff_notes = solid_notes + half_notes + whole_notes
        staff_notes = sorted(staff_notes, key=lambda note: note.box.x)
        all_notes.extend(staff_notes)

    print('[INFO] Converting to MIDI...')
    convert_to_midi(all_notes, name)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
