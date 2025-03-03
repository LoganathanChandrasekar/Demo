app.get("/fetch/userdata/", (req, res) => {
        
     const empId = req.query.empId;   
     const email = req.query.email;  
     const name = req.query.name;
        // console.log(empId, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
      // Create a query object with all three criteria 
     const query = {   empId: empId,   email: email,   name: name };
      console.log("............kkkkkkkkkkkkkkkk",query);
      // Remove undefined fields from query 
     Object.keys(query).forEach(key => {   if (query[key] === undefined) 
     {     delete query[key];   } }); 
     Analyst.find(query)   
     .then((analyst) =>{ res.json(analyst),
     console.log("------>>>>>>>>",analyst)})
        .catch((err) => res.status(400).json("Error: " + err)); 
    });
     
    app.get("/fetch/userdata/", async (req, res) => {
        try {
            const empId = req.query.empId ? String(req.query.empId).trim() : undefined;
            const email = req.query.email ? String(req.query.email).trim().toLowerCase() : undefined;
            const name = req.query.name ? String(req.query.name).trim() : undefined;
    
            // Create a query object with case-insensitive search
            let query = {};
            
            if (empId) {
                query.empId = { $regex: new RegExp('^' + empId + '$', 'i') };
            }
            if (email) {
                query.email = { $regex: new RegExp('^' + email + '$', 'i') };
            }
            if (name) {
                query.name = { $regex: new RegExp('^' + name + '$', 'i') };
            }
    
            console.log("Final Query:", query);
            
            const analyst = await Analyst.findOne(query);
            
            if (!analyst) {
                return res.status(404).json({ message: "No user found with the provided criteria" });
            }
    
            console.log("Query Result:", analyst);
            res.json(analyst);
        } catch (err) {
            console.error("Error:", err);
            res.status(500).json({ error: "Internal server error", details: err.message });
        }
    });


    const express = require('express');
const mongoose = require('mongoose');

app.get("/fetch/userdata/", async (req, res) => {
    try {
        const empId = req.query.empId ? req.query.empId.trim() : undefined;
        const email = req.query.email ? req.query.email.trim() : undefined;
        const name = req.query.name ? req.query.name.trim() : undefined;

        // First, let's find all documents and log their values
        const allDocs = await Analyst.find({});
        console.log("All documents in collection:", allDocs);

        // Try finding with each field individually
        const byEmpId = empId ? await Analyst.find({ empId }) : [];
        const byEmail = email ? await Analyst.find({ email }) : [];
        const byName = name ? await Analyst.find({ name }) : [];

        console.log("Search by empId:", byEmpId);
        console.log("Search by email:", byEmail);
        console.log("Search by name:", byName);

        // Try with OR condition instead of AND
        const query = {
            $or: [
                empId ? { empId } : null,
                email ? { email } : null,
                name ? { name } : null
            ].filter(q => q !== null)
        };

        console.log("Final OR Query:", query);
        const analyst = await Analyst.find(query);
        
        if (analyst.length === 0) {
            // Log the schema of your model
            console.log("Analyst Schema:", Analyst.schema.paths);
            return res.status(404).json({ 
                message: "No user found",
                searchCriteria: { empId, email, name },
                resultsFound: {
                    byEmpId: byEmpId.length,
                    byEmail: byEmail.length,
                    byName: byName.length
                }
            });
        }

        res.json(analyst);
    } catch (err) {
        console.error("Error:", err);
        res.status(500).json({ error: "Internal server error", details: err.message });
    }
});
function formatName(name) {
    if (!name) return ""; // Handle empty or undefined input
    let parts = name.split(" "); // Split by space
    if (parts.length > 1) {
        return parts[0].toLowerCase() + " " + parts.slice(1).join(" "); // Convert first word to lowercase
    }
    return name.toLowerCase(); // If there's only one word, lowercase everything
}
app.get("/fetch/userdata/", (req, res) => {
 
    const named = req.query.name.toLowerCase();  // Convert query to lowercase
    function formatName(named) {
      if (!named) return ""; // Handle empty or undefined input
      let parts = named.split(" "); // Split by space
      if (parts.length > 1) {
          return parts[0].toLowerCase() + " " + parts.slice(1).join(" "); // Convert first word to lowercase
      }
      return named.toLowerCase(); // If there's only one word, lowercase everything
  }
  let formattedName = formatName(named);
   
    Analyst.find({ name: formattedName})
      .then((analyst) => {
        console.log("Query result:", analyst);
        res.json(analyst);
      })
      .catch((err) => {
        console.error("Error:", err);
        res.status(400).json("err" + err);
      });
  });