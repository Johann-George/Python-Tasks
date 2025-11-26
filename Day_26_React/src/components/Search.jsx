import React, { useState } from 'react';
import { TextField, IconButton } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

const SearchBar = ({ onSearch }) => {
    const [searchTerm, setSearchTerm] = useState('')

    const handleInputChange = (e) => {
        setSearchTerm(e.target.value)
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        onSearch(searchTerm)
    };

    return (
        <form onSubmit={handleSubmit} style={{ display: 'flex', alignItems: 'center' }}>
            <TextField
                label="Search"
                variant='outlined'
                size='small'
                value={searchTerm}
                onChange={handleInputChange}
                fullWidth
            />
            <IconButton type='submit' aria-label='search'>
                <SearchIcon/>
            </IconButton>
        </form>
    );
};

export default SearchBar;

