const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const app = express();
const PORT = 3000;

/* =========================
   CONFIG
========================= */
const ROOT_DIR = __dirname;
const PYTHON_DIR = path.join(ROOT_DIR, "python_scripts");
const SAVE_DIR = path.join(ROOT_DIR, "reference_faces");
// Note: we will no longer persist uploaded recognize images to disk.
// The server will stream image bytes to the Python process instead.
const TEMP_DIR = path.join(ROOT_DIR, "temp_recognize");

// Use python from PATH
const PYTHON_CMD = "python";

/* =========================
   ENSURE FOLDERS
========================= */
if (!fs.existsSync(PYTHON_DIR)) {
    console.error("[CRITICAL] Missing python_scripts folder");
    process.exit(1);
}
if (!fs.existsSync(SAVE_DIR)) fs.mkdirSync(SAVE_DIR);
// do not create TEMP_DIR for persistent storage (uploads are in-memory)

/* =========================
   MIDDLEWARE
========================= */
app.use(express.json());
app.use(express.static(ROOT_DIR));

// Serve images
app.use("/reference_faces", express.static(SAVE_DIR));

/* =========================
   MULTER (REGISTER)
========================= */
const registerStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, SAVE_DIR),
    filename: (req, file, cb) => {
        const safeName = (req.body.name || "user")
            .replace(/[^a-z0-9]/gi, "_")
            .toLowerCase();
        cb(null, safeName + path.extname(file.originalname));
    }
});
const uploadRegister = multer({ storage: registerStorage });

/* =========================
   MULTER (RECOGNIZE)
========================= */
// Use memory storage for recognize uploads so we don't save to disk
const tempStorage = multer.memoryStorage();
const uploadTemp = multer({ storage: tempStorage });

/* =========================
   ROUTES
========================= */

// -------- REGISTER --------
app.post("/register", uploadRegister.single("image"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: "No image uploaded" });
    }
    console.log(`[REGISTER] ${req.file.filename}`);
    res.json({ message: "User registered successfully" });
});

// -------- FETCH USERS --------
app.get("/users", (req, res) => {
    fs.readdir(SAVE_DIR, (err, files) => {
        if (err) return res.status(500).json({ error: "Failed to read folder" });

        const users = files
            .filter(f => fs.statSync(path.join(SAVE_DIR, f)).isFile())
            .map((filename, i) => ({
                id: i + 1,
                name: path.parse(filename).name,
                img: `/reference_faces/${filename}`
            }));

        res.json(users);
    });
});
const editStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, SAVE_DIR),
    filename: (req, file, cb) => {
        const oldFilename = req.body.oldFilename;
        if (!oldFilename) return cb(new Error("Missing old filename"));

        const oldExt = path.extname(oldFilename);
        const baseOldName = path.parse(oldFilename).name;

        const finalName = req.body.newName?.trim()
            ? req.body.newName.replace(/[^a-z0-9]/gi, "_").toLowerCase()
            : baseOldName;

        cb(null, finalName + oldExt);
    }
});

const uploadEdit = multer({ storage: editStorage });

// -------- EDIT USER --------
app.put("/edit", uploadEdit.single("image"), (req, res) => {
    const { oldFilename, newName } = req.body;

    if (!oldFilename) {
        return res.status(400).json({ error: "Missing old filename" });
    }

    const oldPath = path.join(SAVE_DIR, oldFilename);
    const oldExt = path.extname(oldFilename);
    const baseOldName = path.parse(oldFilename).name;

    const finalName = newName?.trim()
        ? newName.replace(/[^a-z0-9]/gi, "_").toLowerCase()
        : baseOldName;

    const finalPath = path.join(SAVE_DIR, finalName + oldExt);

    // 🟢 IMAGE UPDATED
    if (req.file) {
        if (oldPath !== finalPath) {
            fs.unlink(oldPath, err => {
                if (err) console.warn("[WARN] Old image delete failed");
            });
        }
        return res.json({ message: "Image updated successfully" });
    }

    // 🟡 NAME ONLY
    if (baseOldName !== finalName) {
        fs.rename(oldPath, finalPath, err => {
            if (err) return res.status(500).json({ error: "Rename failed" });
            return res.json({ message: "Name updated successfully" });
        });
        return;
    }

    // 🔵 NOTHING CHANGED
    res.json({ message: "No changes made" });
});


// -------- DELETE USER --------
app.delete("/delete", (req, res) => {
    const { filename } = req.body;
    if (!filename) return res.status(400).json({ error: "Filename missing" });

    fs.unlink(path.join(SAVE_DIR, filename), err => {
        if (err) return res.status(500).json({ error: "Delete failed" });
        res.json({ message: "Deleted successfully" });
    });
});

// -------- RECOGNIZE --------
app.post("/recognize", uploadTemp.single("image"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ match: false, message: "No image uploaded" });
    }
    const script = "face_recognition_api.py";
    const refRelPath = path.relative(PYTHON_DIR, SAVE_DIR);

    console.log(`[RECOGNIZE] Processing upload ${req.file.originalname}`);

    // Spawn python and tell it to read image bytes from stdin (we use --stdin flag)
    const pythonProcess = spawn(
        PYTHON_CMD,
        [script, "--stdin", refRelPath],
        { cwd: PYTHON_DIR, stdio: ['pipe', 'pipe', 'pipe'] }
    );

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", d => output += d.toString());
    pythonProcess.stderr.on("data", d => errorOutput += d.toString());

    pythonProcess.on("close", code => {
        if (code !== 0) {
            console.error("[PYTHON ERROR]", errorOutput);
            return res.status(500).json({ match: false, error: "Recognition failed" });
        }

        try {
            const jsonMatch = output.match(/\{[\s\S]*\}/);
            if (!jsonMatch) throw new Error("Invalid JSON");

            const result = JSON.parse(jsonMatch[0]);

            // Attach the uploaded image as a data URL so the client can display it
            result.input_img = `data:${req.file.mimetype};base64,${req.file.buffer.toString('base64')}`;

            console.log("[RESULT]", result);
            res.json(result);

        } catch (err) {
            console.error("[PARSE ERROR]", err);
            console.error(output);
            res.status(500).json({ match: false, error: "Invalid AI response" });
        }
    });

    // Write the image bytes to python stdin and close
    pythonProcess.stdin.write(req.file.buffer);
    pythonProcess.stdin.end();
});

/* =========================
   CLEAN TEMP FOLDER
========================= */
app.post("/cleanup-temp", (req, res) => {
    fs.readdir(TEMP_DIR, (err, files) => {
        if (err) {
            console.error("[CLEANUP ERROR]", err);
            return res.status(500).json({ error: "Cleanup failed" });
        }

        files.forEach(file => {
            const filePath = path.join(TEMP_DIR, file);
            fs.unlink(filePath, err => {
                if (err) console.error("[DELETE ERROR]", err);
            });
        });

        console.log("[CLEANUP] temp_recognize wiped");
        res.json({ message: "Temp folder cleaned" });
    });
});

/* =========================
   START SERVER
========================= */
app.listen(PORT, () => {
    console.log("\n--- SYSTEM ONLINE ---");
    console.log(`Server: http://localhost:${PORT}`);
    console.log(`Python CMD: ${PYTHON_CMD}`);
    console.log("---------------------\n");
});
