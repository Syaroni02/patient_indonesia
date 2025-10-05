// Show modal when page loads
    window.onload = function() {
        document.getElementById('loginModal').style.display = "block";
    }

    function closeModal() {
        document.getElementById('loginModal').style.display = "none";
    }

    // Close modal if user clicks outside
    window.onclick = function(event) {
        var modal = document.getElementById('loginModal');
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
