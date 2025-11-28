import React from "react";
import { Typography } from "@mui/material";
import {Table, TableHead, TableBody, TableCell, TableRow, Box} from "@mui/material";

function expandCommaSeparated(row) {
  const values = {
    users_ai: row.users_ai?.split(',') || [],
    users_si: row.users_si?.split(',') || [],
  };

  // determine which column has most rows
  const maxLength = Math.max(values.users_ai.length, values.users_si.length);

  const newRows = [];

  for (let i = 0; i < maxLength; i++) {
    newRows.push({
      users_ai: values.users_ai[i] ?? null,
      users_si: values.users_si[i] ?? null,
      ai_hod: i === 0 ? row.ai_hod : null,
      si_hod: i === 0 ? row.si_hod : null,
      mail_id_ai_missing: i === 0 ? row.mail_id_ai_missing : null,
      si: i === 0 ? row.si : null
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

  return (
    <>
      <Typography variant="h6">Edit Values</Typography>

      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Users AI</TableCell>
            <TableCell>Users SI</TableCell>
            <TableCell>AI HOD</TableCell>
            <TableCell>SI HOD</TableCell>
            <TableCell>Mail ID AI Missing</TableCell>
            <TableCell>SI</TableCell>
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
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </>
  );
}
