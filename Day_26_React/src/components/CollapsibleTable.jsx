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
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

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
    edit: [
      {
        date: '2020-01-05',
        customerId: '11091700',
        amount: 3,
      },
      {
        date: '2020-01-02',
        customerId: 'Anonymous',
        amount: 1,
      },
    ],
  };
}

function Row(props) {
  const { row } = props;
  const [open, setOpen] = React.useState(false);

  return (
    <React.Fragment>
      <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
        <TableCell>
          <IconButton
            aria-label="expand row"
            size="small"
            onClick={() => setOpen(!open)}
          >
            {open ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </TableCell>
        <TableCell component="th" scope="row">
          {row.id}
        </TableCell>
        <TableCell align="right">{row.location}</TableCell>
        <TableCell align="right">{row.user_ai}</TableCell>
        <TableCell align="right">{row.user_si}</TableCell>
        <TableCell align="right">{row.ai_hod}</TableCell>
        <TableCell align="right">{row.si_hod}</TableCell>
        <TableCell align="right">{row.mail_id_ai_missing}</TableCell>
        <TableCell align="right">{row.si}</TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 1 }}>
              <Typography variant="h6" gutterBottom component="div">
                Edit
              </Typography>
              <Table size="small" aria-label="purchases">
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>Location</TableCell>
                    <TableCell align="right">Users_AI</TableCell>
                    <TableCell align="right">Users_SI</TableCell>
                    <TableCell align="right">AI_HOD</TableCell>
                    <TableCell align="right">SI_HOD</TableCell>
                    <TableCell align="right">Mail_ID_AI_Missing</TableCell>
                    <TableCell align="right">SI</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {row.edit.map((editRow) => (
                    <TableRow key={editRow.id}>
                      <TableCell component="th" scope="row">
                        {editRow.location}
                      </TableCell>
                      <TableCell>{editRow.user_ai}</TableCell>
                      <TableCell>{editRow.user_si}</TableCell>
                      <TableCell>{editRow.ai_hod}</TableCell>
                      <TableCell>{editRow.si_hod}</TableCell>
                      <TableCell>{editRow.mail_id_ai_missing}</TableCell>
                      <TableCell>{editRow.si}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </React.Fragment>
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
    mail_id_missing: PropTypes.string.isRequired,
    edit: PropTypes.arrayOf(
      PropTypes.shape({
        location: PropTypes.string.isRequired,
        user_ai: PropTypes.string.isRequired,
        user_si: PropTypes.string.isRequired,
        ai_hod: PropTypes.string.isRequired,
        si_hod: PropTypes.string.isRequired,
        mail_id_missing: PropTypes.string.isRequired,
      }),
    ).isRequired,
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

  const filteredRows = rows.filter((row) => {
    return row.name.toLowerCase().includes(searchQuery.toLowerCase());
  });

  return (
    <TableContainer component={Paper}>
      <Table aria-label="collapsible table">
        <TableHead>
          <TableRow>
            <TableCell />
            <TableCell>Dessert (100g serving)</TableCell>
            <TableCell align="right">Calories</TableCell>
            <TableCell align="right">Fat&nbsp;(g)</TableCell>
            <TableCell align="right">Carbs&nbsp;(g)</TableCell>
            <TableCell align="right">Protein&nbsp;(g)</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {filteredRows.map((row) => (
            <Row key={row.name} row={row} />
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
