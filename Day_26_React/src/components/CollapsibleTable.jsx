import * as React from 'react';
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Collapse from '@mui/material/Collapse';
import IconButton from '@mui/material/IconButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import EditIcon from '@mui/icons-material/Edit'
import EditForm from '../components/EditForm'
import TruncatedCell from './TruncatedCell';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';

function createData(id, location, users_ai, users_si, ai_hod, si_hod, mail_id_ai_missing, si) {
  return {
    id,
    location,
    users_ai,
    users_si,
    ai_hod,
    si_hod,
    mail_id_ai_missing,
    si,
  };
}

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

  for (let i = 0; i < maxLength; i++) {
    newRows.push({
      users_ai: values.users_ai[i] ?? null,
      users_si: values.users_si[i] ?? null,
      ai_hod: values.ai_hod[i] ?? null,
      si_hod: values.si_hod[i] ?? null,
      mail_id_ai_missing: values.mail_id_ai_missing[i] ?? null,
      si: values.si[i] ?? null,
    });
  }

  return newRows;
}

function Row({ row, tableRows, setTableRows }) {

  console.log(row);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editedValues, setEditedValues] = React.useState(null);

  const handleEdit = () => {
    const expandedRows = expandCommaSeparated(row);
    setEditedValues(expandedRows);
    setIsEditing(true)
  }

  const handleSave = () => {

    // const expandedRows = expandCommaSeparated(row);
    // setEditedValues(expandedRows);

    // if(!editedValues || editedValues.length === 0){
    //   setIsEditing(false);
    //   return;
    // }

    const updatedRow = {
      ...row,
      users_ai: editedValues.map(r => r.users_ai).filter(Boolean).join(", "),
      users_si: editedValues.map(r => r.users_si).filter(Boolean).join(", "),
      ai_hod: editedValues.map(r => r.ai_hod).filter(Boolean).join(", "),
      si_hod: editedValues.map(r => r.si_hod).filter(Boolean).join(", "),
      mail_id_ai_missing: editedValues.map(r => r.mail_id_ai_missing).filter(Boolean).join(", "),
      si: editedValues.map(r => r.si).filter(Boolean).join(", "),
    };

    setTableRows(prev => 
      prev.map(r => (r.id === row.id ? updatedRow: r))
    );
    setIsEditing(false);
  };

  return (
    <>
      <TableRow
        sx={{
          '& > *': { borderBottom: 'unset' },
          height: 28,
          '& td, & th': { padding: '4px 8px' }
        }}
      >
        <TableCell component="th" scope="row">{row.id}</TableCell>
        <TableCell>{row.location}</TableCell>
        <TableCell>
          <TruncatedCell text={row.users_ai} maxLength={25}/>
        </TableCell>
        <TableCell>
          <TruncatedCell text={row.users_si} maxLength={25}/>
        </TableCell>
        <TableCell>
          <TruncatedCell text={row.ai_hod} maxLength={25}/>
        </TableCell>
        <TableCell>
          <TruncatedCell text={row.si_hod} maxLength={25}/>
        </TableCell>
        <TableCell>
          <TruncatedCell text={row.mail_id_ai_missing} maxLength={25}/>
        </TableCell>
        <TableCell>
          <TruncatedCell text={row.si} maxLength={25}/>
        </TableCell>
        <TableCell>
          {
            isEditing ? (
              <div>
                <IconButton size='small' onClick={handleSave}>
                  <SaveIcon />
                </IconButton>
                <IconButton size='small' onClick={() => setIsEditing(false)}>
                  <CancelIcon/>
                </IconButton>
              </div>
            ):(
            <IconButton size='small' onClick={handleEdit}>
              <EditIcon />
            </IconButton>
            )
          }
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={9}>
          <Collapse in={isEditing} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <EditForm row={row} onChange={setEditedValues}/>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

Row.propTypes = {
  row: PropTypes.shape({
    id: PropTypes.number.isRequired,
    location: PropTypes.string.isRequired,
    user_ai: PropTypes.string.isRequired,
    user_si: PropTypes.string.isRequired,
    ai_hod: PropTypes.string.isRequired,
    si_hod: PropTypes.string.isRequired,
    mail_id_ai_missing: PropTypes.string.isRequired,
    si: PropTypes.string.isRequired,
  }).isRequired,
};

const rows = [
  createData(30, 'PNQ', 'anju.mallika@quadance.com, anandu.madhu@quadance.com', 'yamini.maa@flyjaclogistics.com,kamalgk@flyjaclogistics.com,alagar.maa@flyjaclogistics.com', 'mohammed.shameer@quadance.com', 'sonia.pnq@flyjaclogistics.com', null, null),
  createData(31, 'HYD', 'anju.mallika@quadance.com, anandu.madhu@quadance.com', 'yamini.maa@flyjaclogistics.com,kamalgk@flyjaclogistics.com,alagar.maa@flyjaclogistics.com', 'mohammed.shameer@quadance.com', 'fjlhyd@flyjaclogistics.com', null, null),
  createData(32, 'COK', 'anju.mallika@quadance.com, anandu.madhu@quadance.com', 'yamini.maa@flyjaclogistics.com,kamalgk@flyjaclogistics.com,alagar.maa@flyjaclogistics.com', 'mohammed.shameer@quadance.com', 'impcord.cok@flyjaclogistics.com', null, null),
  createData(33, 'CJB', 'anju.mallika@quadance.com, anandu.madhu@quadance.com', 'yamini.maa@flyjaclogistics.com,kamalgk@flyjaclogistics.com,alagar.maa@flyjaclogistics.com', 'mohammed.shameer@quadance.com', 'rathnapriya.cbe@flyjaclogistics.com', null, null),
  createData(34, 'GOI', 'anju.mallika@quadance.com, anandu.madhu@quadance.com', 'yamini.maa@flyjaclogistics.com,kamalgk@flyjaclogistics.com,alagar.maa@flyjaclogistics.com', 'mohammed.shameer@quadance.com', 'vithal.goa@flyjaclogistics.com', null, null),
];

export default function CollapsibleTable({ searchQuery }) {

  const [tableRows, setTableRows] = React.useState(rows)

  const filteredRows = tableRows.filter((row) => {
    return row.location.toLowerCase().includes(searchQuery.toLowerCase())
  });

  return (
    <TableContainer component={Paper}>
      <Table 
        aria-label="collapsible table" 
        sx={{
          border:"1px solid #ccc",
          borderRadius: "4px",
          overflow: "hidden"
        }}>
        <TableHead>
          <TableRow>
            <TableCell>ID</TableCell>
            <TableCell>Location</TableCell>
            <TableCell>Users_AI</TableCell>
            <TableCell>Users_SI</TableCell>
            <TableCell>AI_HOD</TableCell>
            <TableCell>SI_HOD</TableCell>
            <TableCell>MAIL_ID_AI_Missing</TableCell>
            <TableCell>SI</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {filteredRows.map((row) => (
            <Row key={row.id} row={row} tableRows={tableRows} setTableRows={setTableRows}/>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
