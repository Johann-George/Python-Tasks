import apiClient from "../api/apiClient";

export async function mailmatrix(){
    try{
        const response = await apiClient.get("/get_data/");
        print("response=",response);
        return response.data;
    }
    catch(error){
        throw new Response(
            error.response?.data?.errorMessage || 
            error.message || 
            "Failed to submit your message. Please try again",
            { status: error.status || 500 }
        );
    }
}

export async function update_mailmatrix({request}){
    try{
        const response = await apiClient.put("/update_data")
    }
}