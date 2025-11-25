import { useState } from 'react'
import StickyHeadTable from './pages/StickyHeadTable'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <StickyHeadTable/>
    </>
  )
}

export default App
