const fs = require('fs');
const path = require('path');

function setupLogger(name) {
    const logFile = path.join(__dirname, '../../../logs', `${name}.log`);
    const log = (level, message) => {
        const timestamp = new Date().toISOString();
        const logMessage = `${timestamp} - ${name} - ${level} - ${message}\n`;
        fs.appendFileSync(logFile, logMessage);
    };
    
    return {
        info: (message) => log('INFO', message),
        error: (message) => log('ERROR', message)
    };
}

module.exports = setupLogger;
