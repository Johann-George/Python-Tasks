import SearchBar from "../components/Search";
import { useState } from "react";
import CollapsibleTable from "../components/CollapsibleTable";
import { useLoaderData } from "react-router-dom";

export default function Data() {

    const [query, setQuery] = useState("");
    const data = useLoaderData();
    // const [tableData, setTableData] = useState(rows)
    const handleSearch = (searchTerm) => {
        console.log("Search query:", searchTerm)
        setQuery(searchTerm)
    };
    
    const handleEdit = (updatedRow) => {
        setTableData((prev) => 
            prev.map((row) => 
                row.code === updatedRow.code ? updatedRow : row
            )
        );
    }; 

    return (
        <>
            <SearchBar onSearch={handleSearch}/>
            <CollapsibleTable
                searchQuery={query}
                data = {data}
            />
            {/* <StickyHeadTable 
                searchQuery={query}
                tableData={tableData}    
                onEdit={handleEdit}
            /> */}
        </>
    );
}