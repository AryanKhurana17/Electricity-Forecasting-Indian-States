import React, { useState, useEffect } from "react";
import {
  Container, Typography, Box, FormControl, InputLabel, Select, MenuItem, Button, Alert, CircularProgress, TextField, Paper
} from "@mui/material";
import { Line } from "react-chartjs-2";

const API_BASE = "http://localhost:8000";

const stateOptions = [
  "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "DD",
  "Delhi", "DNH", "DVC", "Essar steel", "Goa", "Gujarat", "Haryana", "HP", "J&K", "Jharkhand",
  "Karnataka", "Kerala", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "MP", "Nagaland",
  "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "UP", "Uttarakhand",
  "West Bengal", "Total Consumption"
];

function App() {
  const [state, setState] = useState("");
  const [date, setDate] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState({});
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/data-info/`)
      .then(res => res.json())
      .then(data => setInfo(data))
      .catch(() => setInfo({ last_data_date: "N/A" }));
    setDate(new Date().toISOString().split('T')[0]);
  }, []);

  const handlePredict = () => {
    setLoading(true);
    setResult(null);
    setError("");
    fetch(`${API_BASE}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state_name: state, target_date: date }),
    })
      .then(async res => {
        if (!res.ok) {
          const data = await res.json();
          throw new Error(data.detail || "Prediction Error");
        }
        return res.json();
      })
      .then(data => setResult(data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <Container maxWidth="sm">
      <Box mt={4} mb={2} component={Paper} p={3} elevation={3}>
        <Typography variant="h4" align="center" gutterBottom>
          🔋 Electricity Consumption Forecaster
        </Typography>
        <Typography align="center" color="textSecondary" mb={2}>
          Data available until: <b>{info.last_data_date}</b>
        </Typography>
        <FormControl fullWidth margin="normal">
          <InputLabel>State/Region</InputLabel>
          <Select value={state} onChange={e => setState(e.target.value)}>
            {stateOptions.map(o => <MenuItem key={o} value={o}>{o}</MenuItem>)}
          </Select>
        </FormControl>
        <TextField
          label="Target Date"
          type="date"
          InputLabelProps={{ shrink: true }}
          value={date}
          onChange={e => setDate(e.target.value)}
          fullWidth
          margin="normal"
        />
        <Box mt={2} display="flex" gap={1}>
          <Button
            type="button"
            variant="contained"
            color="primary"
            fullWidth
            onClick={handlePredict}
            disabled={!state || !date || loading}
          >Predict</Button>
        </Box>
        {loading && <Box textAlign="center" my={2}><CircularProgress /></Box>}
        {result &&
          <Alert severity="success" sx={{ mt: 2 }}>
            <Typography><b>{result.state_name}</b>, {result.target_date}:</Typography>
            <Typography variant="h5" color="primary" sx={{ mt: 1 }}>
              {result.predicted_consumption.toFixed(2)} MW
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              Days ahead: {result.days_ahead}<br />
              Base data date: {result.last_data_date}
            </Typography>
          </Alert>
        }
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </Box>
      {/* Optionally, visualize time series */}
      {/* <Box my={3}>
        <Typography variant="h6">Trend for {state} (Coming Soon...)</Typography>
        <Line data={...} />
      </Box> */}
    </Container>
  );
}

export default App;
