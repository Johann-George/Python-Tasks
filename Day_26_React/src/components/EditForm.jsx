import React from "react";
import { Button, IconButton, Typography } from "@mui/material";
import {Table, TableHead, TableBody, TableCell, TableRow, Box} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete"

function expandCommaSeparated(row) {
  const values = {
    users_ai: row.users_ai?.split(',') || [],
    users_si: row.users_si?.split(',') || [],
    ai_hod: row.ai_hod?.split(',') || [],
    si_hod: row.si_hod?.split(',') || [],
    mail_id_ai_missing: row.mail_id_ai_missing?.split(',') || [],
    si: row.si?.split(',') || [],
  };

  // determine which column has most rows
  const maxLength = Math.max(values.users_ai.length, values.users_si.length, values.ai_hod.length, values.si_hod.length, values.mail_id_ai_missing.length, values.si.length);

  const newRows = [];

  for (let i = 0; i < maxLength; i++) {
    newRows.push({
      users_ai: values.users_ai[i] ?? null,
      users_si: values.users_si[i] ?? null,
      ai_hod: values.ai_hod[i] ?? null,
      si_hod: values.si_hod[i] ?? null,
      mail_id_ai_missing: values.mail_id_ai_missing[i] ?? null,
      si: values.si[i] ?? null
    });
  }

  return newRows;
}

export default function EditForm({ row, onChange }) {
  const expandedRows = expandCommaSeparated(row);
  const [formRows, setFormRows] = React.useState(expandedRows);

  const handleChange = (index, field, value) => {
    const updated = [...formRows];
    updated[index][field] = value;
    setFormRows(updated);

    onChange(updated)
  };

  const handleAddRow = () => {
    const newRow = {
      users_ai: "",
      users_si: "",
      ai_hod: "",
      si_hod: "",
      mail_id_ai_missing: "",
      si: "",
    }

    const updated = [...formRows, newRow];
    setFormRows(updated)
    onChange(updated)
  }

  const handleDeleteRow = (index) => {
    const updated = formRows.filter((_, i) => i !== index);
    setFormRows(updated);
    onChange(updated)
  }

  return (
    <>
      <Typography variant="h6">Edit Values</Typography>
      <Button variant="outlined" size="small" onClick={handleAddRow}>
        Add Row
      </Button>
      
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Users AI</TableCell>
            <TableCell>Users SI</TableCell>
            <TableCell>AI HOD</TableCell>
            <TableCell>SI HOD</TableCell>
            <TableCell>Mail ID AI Missing</TableCell>
            <TableCell>SI</TableCell>
            <TableCell>Delete</TableCell>
          </TableRow>
        </TableHead>

        <TableBody>
          {formRows.map((r, index) => (
            <TableRow key={index}>
              <TableCell>
                <input
                  value={r.users_ai ?? ''}
                  onChange={(e) => handleChange(index, 'users_ai', e.target.value)}
                />
              </TableCell>

              <TableCell>
                <input
                  value={r.users_si ?? ''}
                  onChange={(e) => handleChange(index, 'users_si', e.target.value)}
                />
              </TableCell>

              <TableCell>
                <input
                  value={r.ai_hod ?? ''}
                  onChange={(e) => handleChange(index, 'ai_hod', e.target.value)}
                />
              </TableCell>

              <TableCell>
                <input
                  value={r.si_hod ?? ''}
                  onChange={(e) => handleChange(index, 'si_hod', e.target.value)}
                />
              </TableCell>

              <TableCell>
                <input
                  value={r.mail_id_ai_missing ?? ''}
                  onChange={(e) =>
                    handleChange(index, 'mail_id_ai_missing', e.target.value)
                  }
                />
              </TableCell>

              <TableCell>
                <input
                  value={r.si ?? ''}
                  onChange={(e) => handleChange(index, 'si', e.target.value)}
                />
              </TableCell>
              <TableCell>
                <IconButton size="small" color="error" onClick={() => handleDeleteRow(index)}>
                  <DeleteIcon/>
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      
    </>
  );
}
