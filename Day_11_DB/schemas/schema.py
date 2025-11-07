from pydantic import BaseModel, Field, EmailStr, validator

class OrderInput(BaseModel):
    item_name: str = Field(..., min_length=1)
    item_price: float = Field(..., gt=0)
    item_quantity: int = Field(..., gt=0)

class CustomerInput(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=10)

    @validator("name")
    def validate_name(cls, value):
        if not value.replace(" ", "").isalpha():
            raise ValueError("Name must contains only letters and spaces.")
        return value

    @validator("phone")
    def validate_phone(cls, value):
        if not value.isdigit():
            raise ValueError("Phone number must contain digits only")
        return value
