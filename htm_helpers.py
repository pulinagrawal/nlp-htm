def get_columns_from_cells(tm, cells):
    columns = set()
    for cell in cells:
        columns.add(tm.columnForCell(cell))
    return list(columns)
