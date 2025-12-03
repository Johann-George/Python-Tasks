import apiClient from "../api/apiClient";

export async function mailmatrix(){
    try{
        const data = await apiClient.get("/get_data/");
        return data;
    }
    catch(error){
        if(error.status === 400){
            return {
                success: false,
                errors: error.response?.data
            }
        }
        throw new Response(
            error.response?.data?.error || "Failed to submit your message. Please try again",
            { status: error.status || 500 }
        )
    }
}