import React from "react";
import { Typography } from "@mui/material";
import { Table, TableHead, TableBody, TableCell, TableRow, Box } from "@mui/material";

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
            id: i === 0 ? row.id : null,
            location: i === 0 ? row.location : null,
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

export default function ExpandedData({ row }) {

    const expandedRows = expandCommaSeparated(row);

    return (
        <>
            <Typography variant="h6" gutterBottom component="div">
                Edit
            </Typography>
            <Table size="small" aria-label="purchases">
                <TableHead>
                    <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Location</TableCell>
                        <TableCell>Users AI</TableCell>
                        <TableCell>Users SI</TableCell>
                        <TableCell>AI HOD</TableCell>
                        <TableCell>SI HOD</TableCell>
                        <TableCell>Mail ID AI Missing</TableCell>
                        <TableCell>SI</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {expandedRows.map((editRow, index) => (
                        <TableRow key={index}>
                            <TableCell >{editRow.id}</TableCell>
                            <TableCell>
                                {editRow.location}
                            </TableCell>
                            <TableCell>{editRow.users_ai}</TableCell>
                            <TableCell>{editRow.users_si}</TableCell>
                            <TableCell>{editRow.ai_hod}</TableCell>
                            <TableCell>{editRow.si_hod}</TableCell>
                            <TableCell>{editRow.mail_id_ai_missing}</TableCell>
                            <TableCell>{editRow.si}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </>
    )
}