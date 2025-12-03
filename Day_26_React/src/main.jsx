import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { createBrowserRouter, createRoutesFromElements, Route, RouterProvider } from 'react-router-dom'
import App from './App.jsx'
import Data from './pages/Data.jsx'
import { mailmatrix } from './pages/mailmatrix.js'

const routeDefinitions = createRoutesFromElements(
  <Route path='/' element={<App/>}>
    <Route index element={<Data/>} action={mailmatrix}/>
  </Route>
)

const appRouter = createBrowserRouter(routeDefinitions);

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={appRouter} />
  </StrictMode>,
)
