const express = require('express');
const app = express();
const multer = require('multer');
const upload = multer({ dest: './uploads/' });

app.use(express.json());

app.post('/predict', upload.array('files', 3), (req, res) => {
    const files = req.files;
    // Process files and make prediction using Jupyter notebook
    const prediction = // call Jupyter notebook API or run Python script to make prediction
    res.json({ prediction });
});

app.listen(3000, () => {
    console.log('Server started on port 3000');
});