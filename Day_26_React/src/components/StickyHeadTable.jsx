import * as React from 'react';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TablePagination from '@mui/material/TablePagination';
import TableRow from '@mui/material/TableRow';
import IconButton from '@mui/material/IconButton';
import EditIcon from "@mui/icons-material/Edit"
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import TextField from '@mui/material/TextField';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';

const columns = [
  { id: 'name', label: 'Name', minWidth: 170 },
  { id: 'code', label: 'ISO\u00a0Code', minWidth: 100 },
  {
    id: 'population',
    label: 'Population',
    minWidth: 170,
    // align: 'right',
    format: (value) => value.toLocaleString('en-US'),
  },
  {
    id: 'size',
    label: 'Size\u00a0(km\u00b2)',
    minWidth: 170,
    // align: 'right',
    format: (value) => value.toLocaleString('en-US'),
  },
  {
    id: 'density',
    label: 'Density',
    minWidth: 170,
    // align: 'right',
    format: (value) => value.toFixed(2),
  },
  {
    id: 'actions',
    label: 'Actions',
    minWidth: 100
  }
];


export default function StickyHeadTable({ searchQuery, tableData, onEdit }) {

  const [open, setOpen] = React.useState(false);
  const [editRow, setEditRow] = React.useState(null);

  const handleOpen = (row) => {
    setEditRow({ ...row });
    setOpen(true);
  }

  const handleClose = () => setOpen(false);

  const handleSave = (e) => {
    onEdit(editRow);
    setOpen(false)
  }

  const handleChange = (e) => {
    setEditRow({
      ...editRow,
      [e.target.name]: e.target.value
    });
  };

  const filteredRows = tableData.filter((row) => {
    return row.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      row.code.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(+event.target.value);
    setPage(0);
  };

  return (
    <>
      <Paper sx={{ width: '100%', overflow: 'hidden' }}>
        <TableContainer sx={{ maxHeight: 440 }}>
          <Table stickyHeader aria-label="sticky table">
            <TableHead>
              <TableRow>
                {columns.map((column) => (
                  <TableCell
                    key={column.id}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                  >
                    {column.label}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredRows
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row) => {
                  return (
                    <TableRow hover role="checkbox" tabIndex={-1} key={row.code}>
                      {columns.map((col) => (
                        col.id !== "actions" ? (
                          <TableCell key={col.id}>
                            {row[col.id]}
                          </TableCell>
                        ) : (
                          <TableCell key="actions">
                            <IconButton onClick={() => handleOpen(row)}>
                              <EditIcon />
                            </IconButton>
                          </TableCell>
                        )
                      ))}
                    </TableRow>
                  );
                })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[10, 25, 100]}
          component="div"
          count={filteredRows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
      {/* Edit Dialog */}
      <Dialog open={open} onClose={handleClose}>
        <DialogTitle>Edit Row</DialogTitle>
        <DialogContent>

          <TextField
            margin="dense"
            label="Name"
            name="name"
            value={editRow?.name || ""}
            onChange={handleChange}
            fullWidth
          />

          <TextField
            margin="dense"
            label="Code"
            name="code"
            value={editRow?.code || ""}
            onChange={handleChange}
            fullWidth
          />

          <TextField
            margin="dense"
            label="Population"
            name="population"
            type="number"
            value={editRow?.population || ""}
            onChange={handleChange}
            fullWidth
          />

          <TextField
            margin="dense"
            label="Size"
            name="size"
            type="number"
            value={editRow?.size || ""}
            onChange={handleChange}
            fullWidth
          />

        </DialogContent>

        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button onClick={handleSave} variant="contained">Save</Button>
        </DialogActions>

      </Dialog>
    </>
  );
}
