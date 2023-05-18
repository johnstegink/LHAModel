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
    /* Set the hovering on */
    $(".section").click( clickSection )
})



/*
    Show the arrows of the given relations
 */

function showArrows( relations, leftToRight)
{
    for( relation of relations) {
        let src =  leftToRight ? $("#" + relation.src)[0] : $("#" + relation.dest)[0] ;
        let dest = leftToRight ? $("#" + relation.dest)[0] : $("#" + relation.src)[0];

        let similarity = Math.round( relation.similarity * 100);

        line = new LeaderLine( src, dest, {
            middleLabel: similarity + "%",
            size: ((similarity -50) / 50 + 1) * 2,
            color: "#888888",
            startSocket: leftToRight ? "right" : "left",
            endSocket: leftToRight ? "left" : "right",

        });
        gShownLines.push( line);
    }
}


/*
    Hide all arrows in gShowLines
 */
function hideArrows()
{
    for( line of gShownLines)
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
    let lineId = "line_" + (gRelations.length + 1)
    gRelations.push({ src: srcId, dest: destId, similarity: similarity, lineId: lineId })
}



/*
    A click was done on the section
 */
function clickSection()
{
    let element = $(this);
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
    let id = element.attr('id');
    let srcToDest = isSourceElement(id);

    addRemoveClass( element,"source_selected", show );

    relations = getRelations( id);
    for( relation of relations)
    {
        let src  = $("#" + (srcToDest ? relation.src : relation.dest));
        let dest = $("#" + (srcToDest ? relation.dest : relation.src));

        addRemoveClass( dest,"destination_selected", show );
    }

    hideArrows();       // Allways hide the lines
    if( show) {
        showArrows( relations, srcToDest);
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
    let relations = []
    for( relation of gRelations){
        if( relation.src == src  ||  relation.dest == src) {
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
    for( relation of gRelations) {
        if (relation.src == id) {  /* It is a source */
            return true;
        }
    }

    return false; /* No source found, it is a destination */
}