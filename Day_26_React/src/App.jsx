import './App.css'
import { useNavigation, Outlet } from 'react-router-dom'
import CircularProgress from '@mui/material/CircularProgress';
import Box from '@mui/material/Box';

function App() {

  const navigation = useNavigation();
  return (
    <>
      {navigation.state === "loading" ? (
        <Box sx={{ display: 'flex' }}>
          <CircularProgress />
        </Box>
      ) : (
        <Outlet />
      )}
    </>
  )
}

export default App
