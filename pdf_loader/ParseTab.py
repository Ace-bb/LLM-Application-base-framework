"""
Return a Python list of lists from the words found in a fitz.Document page
-------------------------------------------------------------------------------
License: GNU GPL V3
(c) 2018 Jorj X. McKie

Usage
-----
Used by extract.py and wx-extract.py

Notes
-----
(1) Works correctly for simple, non-nested tables only.

(2) Line recognition depends on the coordinates of the detected words in the
    rectangle. These will be round to integer (pixel) values. However, use of
    different fonts, scan inaccuracies, and so on, may lead to artefacts line
    differences, which must be handled by the caller.

Dependencies
-------------
PyMuPDF v1.12.0 or later
"""
from conf import *
from operator import itemgetter
from itertools import groupby
import fitz


# ==============================================================================
# Function ParseTab - parse a document table into a Python list of lists
# ==============================================================================
def ParseTab(page, bbox, columns=None):
    """Returns the parsed table of a page in a PDF / (open) XPS / EPUB document.
    Parameters:
    page: fitz.Page object
    bbox: containing rectangle, list of numbers [xmin, ymin, xmax, ymax]
    columns: optional list of column coordinates. If None, columns are generated
    Returns the parsed table as a list of lists of strings.
    The number of rows is determined automatically
    from parsing the specified rectangle.
    """
    tab_rect = fitz.Rect(bbox).irect
    xmin, ymin, xmax, ymax = tuple(tab_rect)

    if tab_rect.is_empty or tab_rect.is_infinite:
        print("Warning: incorrect rectangle coordinates!")
        return []

    if type(columns) is not list or columns == []:
        coltab = [tab_rect.x0, tab_rect.x1]
    else:
        coltab = sorted(columns)

    if xmin < min(coltab):
        coltab.insert(0, xmin)
    if xmax > coltab[-1]:
        coltab.append(xmax)

    words = page.get_text("words")

    if words == []:
        print("Warning: page contains no text")
        return []

    alltxt = []

    # get words contained in table rectangle and distribute them into columns
    for w in words:
        ir = fitz.Rect(w[:4]).irect  # word rectangle
        if ir in tab_rect:
            cnr = 0  # column index
            for i in range(1, len(coltab)):  # loop over column coordinates
                if ir.x0 < coltab[i]:  # word start left of column border
                    cnr = i - 1
                    break
            alltxt.append([ir.x0, ir.y0, ir.x1, cnr, w[4]])

    if alltxt == []:
        print("Warning: no text found in rectangle!")
        return []

    alltxt.sort(key=itemgetter(1))  # sort words vertically

    # create the table / matrix
    spantab = []  # the output matrix

    for y, zeile in groupby(alltxt, itemgetter(1)):
        schema = [""] * (len(coltab) - 1)
        for c, words in groupby(zeile, itemgetter(3)):
            entry = " ".join([w[4] for w in words])
            schema[c] = entry
        spantab.append(schema)

    return spantab

def ParseTab_v2(doc, page, bbox, columns = None):
    ''' Returns the parsed table of a page in a PDF / (open) XPS / EPUB document.
    Parameters:
    doc: a fitz.Document
    page: integer page number (0-based)
    bbox: containing rectangle, list of numbers [xmin, ymin, xmax, ymax]
    columns: optional list of column coordinates. If None, columns are generated.

    Returns the parsed table as a list of lists of strings.
    '''
    import json
    import sqlite3
    xmin, ymin, xmax, ymax = bbox                # rectangle coordinates
    if not (xmin < xmax and ymin < ymax):
        print("Warning: incorrect rectangle coordinates!")
        return []

    if type(page) == type(1):
        txt = doc.getPageText(page, output="json") # page text in JSON format
    else:
        txt = page.getText(output = "json")

    blocks = json.loads(txt)["blocks"]             # get list of blocks
    if not blocks:
        print("Warning: page contains no text")
        return []
    db = sqlite3.connect(":memory:")        # create RAM database
    cur = db.cursor()
    # create a table for the spans (text pieces)
    cur.execute("CREATE TABLE `spans` (`x0` REAL,`y0` REAL, `text` TEXT)")

#==============================================================================
#   Function spanout - store a span in database
#==============================================================================
    def spanout(s, y0):
        x0  = s["bbox"][0]
        txt = s["text"]          # the text piece
        cur.execute("insert into spans values (?,?,?)", (int(x0), int(y0), txt))
        return
#==============================================================================
    # populate database with all spans in the requested bbox
    for block in blocks:
        for line in block["lines"]:
            y0 = line["bbox"][1]            # top-left y-coord
            y1 = line["bbox"][3]            # bottom-right y-coord
            if y0 < ymin or y1 > ymax:      # line outside bbox limits - skip it
                continue
            spans = []                      # sort spans by their left coord's
            for s in line["spans"]:
                if s["bbox"][0] >= xmin and s["bbox"][2] <= xmax:
                    spans.append([s["bbox"][0], s])
            if spans:                       # any spans left at all?
                spans.sort()                # sort them
            else:
                continue
            # concatenate spans close to each other
            for i, s in enumerate(spans):
                span = s[1]
                if i == 0:
                    s0 = span                    # memorize 1st span
                    continue
                x0  = span["bbox"][0]            # left borger of span
                x1  = span["bbox"][2]            # right border of span
                txt = span["text"]               # text of this span
                if abs(x0 - s0["bbox"][2]) > 3:  # if more than 3 pixels away
                    spanout(s0, y0)              # from previous span, output it
                    s0 = span                    # store this one as new 1st
                    continue
                s0["text"] += txt                # join current span with prev
                s0["bbox"][2] = x1               # save new right border
            spanout(s0, y0)                      # output the orphan

    # create a list of all the begin coordinates (used for column indices).

    if columns:                        # list of columns provided by caller
        coltab = columns
        coltab.sort()                  # sort it to be sure
        if coltab[0] > xmin:
            coltab = [xmin] + coltab   # left rect border is a valid delimiter
    else:
        cur.execute("select distinct x0 from spans order by x0")
        coltab = [t[0] for t in cur.fetchall()]

    # now read all text pieces from top to bottom.
    cur.execute("select x0, y0, text from spans order by y0")
    alltxt = cur.fetchall()
    db.close()                              # do not need database anymore

    # create the matrix
    spantab = []

    try:
        y0 = alltxt[0][1]                   # y-coord of first line
    except IndexError:                      # nothing there:
        print("Warning: no text found in rectangle!")
        return []

    zeile = [""] * len(coltab)

    for c in alltxt:
        c_idx = len(coltab) - 1
        while c[0] < coltab[c_idx]:         # col number of the text piece
            c_idx = c_idx - 1
        if y0 < c[1]:                       # new line?
            # output old line
            spantab.append(zeile)
            # create new line skeleton
            y0 = c[1]
            zeile = [""] * len(coltab)
        if not zeile[c_idx] or zeile[c_idx].endswith(" ") or\
                               c[2].startswith(" "):
            zeile[c_idx] += c[2]
        else:
            zeile[c_idx] += " " + c[2]

    # output last line
    spantab.append(zeile)
    return spantab

class TableLoader:
    def __init__(self, pdf_file, flavor="lattice") -> None:
        self.pdf_file = pdf_file
        self.flavor = flavor
        self.tables = list()
        self.first_word = None
        self.last_word = None

    def exist_table(self, page_num:list):
        tables = camelot.read_pdf(self.pdf_file, pages=int_list_2_str(page_num))
        if tables.__len__() == 0:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                tables = camelot.read_pdf(self.pdf_file, pages=int_list_2_str(page_num), flavor='stream')
                if len(w) > 0:
                    return False
            if tables.__len__() == 0:
                return False
            else:
                self.flavor = 'stream'
                return True
        else:
            return True
            
    
    def load_table(self, page_num:list = [1], page_text:str = None):
        tables = camelot.read_pdf(self.pdf_file, pages=int_list_2_str(page_num), flavor=self.flavor)
            
        if len(tables) <= 0 : return None, None
        table = tables[0]
        print(f"This table's accuracy is {table.accuracy}")
        if table.accuracy < 90:
            if self.flavor == 'stream':
                return None, None
            self.flavor = 'stream'
            return self.load_table(page_num=page_num, page_text=page_text)
        table_df = table.df
        table_data = ''
        for i in range(table_df.index.size):
            row = '| '
            for j in range(table_df.columns.size):
                page_text.replace(table_df.loc[i,j], '')
                # if self.first_word == None and table_df.loc[i,j] != '' and '表' not in table_df.loc[i,j] and table_df.loc[i,j] in page_text:
                #     self.first_word = table_df.loc[i,j]
                # if table_df.loc[i,j] != '' and table_df.loc[i,j] in page_text: self.last_word = table_df.loc[i,j]

                row += f"{table_df.loc[i,j]} | "
            table_data += f"{row}\n"
        console.log(f"[green]{table_data}")
        self.tables.append(table_data)

        return table_data, page_text
    
    def table_split(self):
        ...