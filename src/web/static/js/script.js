// JavaScript for Fraud Detection System

// Function to generate a random transaction number
function generateTransactionNumber() {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15);
}

// Function to fill form with sample data
function fillSampleData() {
    // Sample transaction data
    const now = new Date();
    const formattedDate = now.toISOString().slice(0, 16);
    
    document.getElementById('trans_date_trans_time').value = formattedDate;
    document.getElementById('cc_num').value = '4532' + Math.floor(1000000000000 + Math.random() * 9000000000000);
    document.getElementById('merchant').value = 'Sample Merchant';
    document.getElementById('category').value = 'shopping_pos';
    document.getElementById('amt').value = (Math.random() * 1000).toFixed(2);
    document.getElementById('first').value = 'John';
    document.getElementById('last').value = 'Doe';
    document.getElementById('gender').value = 'M';
    document.getElementById('dob').value = '1980-01-01';
    document.getElementById('job').value = 'Software Developer';
    document.getElementById('street').value = '123 Main St';
    document.getElementById('city').value = 'New York';
    document.getElementById('state').value = 'NY';
    document.getElementById('zip').value = '10001';
    document.getElementById('lat').value = '40.7128';
    document.getElementById('long').value = '-74.0060';
    document.getElementById('city_pop').value = '8336817';
    document.getElementById('merch_lat').value = '40.7128';
    document.getElementById('merch_long').value = '-74.0060';
    document.getElementById('trans_num').value = generateTransactionNumber();
    document.getElementById('unix_time').value = Math.floor(now.getTime() / 1000);
}

// Add event listener when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add sample data button if it exists
    const sampleDataBtn = document.getElementById('sample-data-btn');
    if (sampleDataBtn) {
        sampleDataBtn.addEventListener('click', fillSampleData);
    }
    
    // Set current date and time as default for transaction time
    const transDateTimeInput = document.getElementById('trans_date_trans_time');
    if (transDateTimeInput) {
        const now = new Date();
        const formattedDate = now.toISOString().slice(0, 16);
        transDateTimeInput.value = formattedDate;
    }
    
    // Set current unix time as default
    const unixTimeInput = document.getElementById('unix_time');
    if (unixTimeInput && !unixTimeInput.value) {
        const now = new Date();
        unixTimeInput.value = Math.floor(now.getTime() / 1000);
    }
    
    // Generate random transaction number if empty
    const transNumInput = document.getElementById('trans_num');
    if (transNumInput && !transNumInput.value) {
        transNumInput.value = generateTransactionNumber();
    }
});
