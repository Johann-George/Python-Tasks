import { useState } from 'react'
import StickyHeadTable from './components/StickyHeadTable.jsx'
import Grid from '@mui/material/Grid'
import Data from './pages/Data.jsx'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <Data /> 
    </>
  )
}

export default App
