/* javascript file that shows the linking of the sections */


// All relations
gRelations = []
gPreviouseSelectedId = ""
gShownLines = []

/*
    Initialization
 */
$(document).ready( () =>
{
    markActiveSelections();
    /* Set the hovering on */
    $(".section").click( clickSection )
})



/*
 * Mark all sections that are active
 */
function markActiveSelections()
{
    for( const id of getIdsFromRelations( gRelations))
    {
        const element = $("#" + id);
        element.addClass("activeSection");
    }
}

/*
    Show the arrows of the given relations
 */

function showArrows( relations)
{
    for( const relation of relations) {
        const src =  $("#" + relation.src)[0];
        const dest = $("#" + relation.dest)[0];

        const a = 1
        const similarity = Math.round( relation.similarity * 100);

        const line = new LeaderLine( src, dest, {
            middleLabel: similarity + "%",
            size: ((similarity -50) / 50 + 1) * 2,
            color: "#888888",
            startSocket: "right",
            endSocket: "left",

        });
        gShownLines.push( line);
    }
}


/*
    Hide all arrows in gShowLines
 */
function hideArrows()
{
    for( const line of gShownLines)
    {
        line.remove();
    }
    gShownLines = [];  // All are hidden
}


/*
    Adds a relation between the source and the destination
 */
function add_relation( srcId, destId, similarity)
{
    const lineId = "line_" + (gRelations.length + 1)
    gRelations.push({ src: srcId, dest: destId, similarity: similarity, lineId: lineId })
}



/*
    A click was done on the section
 */
function clickSection()
{
    const element = $(this);
    /* Hide the previouse selected section */
    if( gPreviouseSelectedId != "") {
        hideShow( $("#" + gPreviouseSelectedId), false);
    }
    hideShow( element, true);
    gPreviouseSelectedId = element.attr('id');
}


/*
    Hides or shows an element, show is a boolean
 */
function hideShow( element, show)
{
    const id = element.attr('id');

    addRemoveClass( element,"source_selected", show );

    relations = getRelations( id);
    for( const relation of relations)
    {
        const src  = $("#" + relation.src);
        const dest = $("#" + relation.dest);

        addRemoveClass( dest,"destination_selected", show );
    }

    hideArrows();       // Allways hide the lines
    if( show) {
        showArrows( relations);
    }
}



/*
    Adds or removes a class
 */
function addRemoveClass( element, cssClass, add)
{
    if( add){
        element.addClass(cssClass)
    }
    else {
        element.removeClass(cssClass)
    }
}

/*
    returns a list of all destinations of this source
 */
function getRelations( src)
{
    const relations = []
    for( const relation of gRelations){
        if( relation.src == src ) {
            relations.push( relation)
        }
    }

    return relations
}


/*
    Returns true if the element is a source to destination element, false otherwise
 */
function isSourceElement( id)
{
    for( const relation of gRelations) {
        if (relation.src == id) {  /* It is a source */
            return true;
        }
    }

    return false; /* No source found, it is a destination */
}


/*
    Returns a set with all ids from the relations, both source and destination are being considered
 */
function getIdsFromRelations( relations)
{
    const ids = new Set()
    for( const relation of gRelations) {
        ids.add( relation.src);
    }

    return Array.from(ids);
}