const fs = require('fs');
const path = require('path');
const setupLogger = require('../../src/js/api/utils/logging');

describe('Logging Tests', () => {
    it('should log info message to file', () => {
        const logger = setupLogger('test');
        logger.info('Test message');
        
        const logFile = path.join(__dirname, '../../logs', 'test.log');
        const content = fs.readFileSync(logFile, 'utf8');
        expect(content).toMatch(/INFO - Test message/);
        
        fs.unlinkSync(logFile);
    });
});
