project Introduces feature introduces an improved shopping flow in the Pluto application. When a user clicks "Add to Cart" on any product recommendation, the application automatically redirects them to a dedicated Checkout page. This eliminates unnecessary steps and provides the user with immediate feedback.

What Happens After an Item Is Added to the Cart

The selected product is stored in the cart.

The application automatically navigates to the new "Checkout" page.

The Checkout page displays the following information:

Product name

Product price

Expected delivery period (5–7 days)

Payment method (default: Credit Card)

Order date and timestamp

A "Place Order" button is available for the user to confirm and complete the purchase.

Purpose of the Feature

Reduces the number of interactions required to review item details.

Provides an immediate and clear confirmation after adding an item to the cart.

Creates a streamlined and intuitive shopping flow similar to real-world e-commerce applications.

Helps users proceed from selection to checkout more efficiently.

Technical Overview

A new Checkout page has been added to the app using a dedicated conditional block:
elif st.session_state.page == "Checkout":

The most recently added item is stored in st.session_state.last_added.

The redirection is handled by setting
st.session_state.page = "Checkout"
followed by st.rerun() after the item is added.

Existing functionalities and UI components remain unchanged; the feature only adds a new flow after the "Add to Cart" action.

