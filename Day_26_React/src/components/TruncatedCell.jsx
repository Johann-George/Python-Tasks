import { Tooltip, Typography } from "@mui/material";

const TruncatedCell = ({ text, maxLength = 30 }) => {
  const truncated =
    text && text.length > maxLength ? text.substring(0, maxLength) + "..." : text;

  return (
    <Tooltip title={text} placement="top" arrow>
      <Typography
        variant="body2"
        sx={{
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          maxWidth: 200,  // adjust width based on your table
          cursor: "pointer"
        }}
      >
        {truncated}
      </Typography>
    </Tooltip>
  );
};

export default TruncatedCell;
