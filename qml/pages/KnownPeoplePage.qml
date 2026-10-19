pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "knownPeoplePage"
    property var people: null
    property var inspector: null
    property string viewSize: "Medium"
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    readonly property bool detailOpen: people !== null && Boolean(people.selectedKnown.key)

    ColumnLayout {
        anchors.fill: parent
        visible: !root.detailOpen
        spacing: 12
        RowLayout {
            Layout.fillWidth: true
            TextField {
                id: knownSearch
                objectName: "knownPeopleSearch"
                Layout.preferredWidth: Math.min(320, root.width * 0.45)
                implicitHeight: 42
                placeholderText: "Search names"
                text: root.people ? root.people.knownModel.searchQuery : ""
                onTextEdited: if (root.people) root.people.knownModel.setSearch(text)
                Accessible.name: "Search known people by name"
                background: Rectangle {
                    radius: height / 2
                    color: theme.panelRaisedFill
                    border.width: knownSearch.activeFocus ? 2 : 1
                    border.color: knownSearch.activeFocus ? theme.focusRing : theme.separator
                }
            }
            Components.ToolbarDropdownButton {
                objectName: "knownPeopleSort"
                label: root.people ? root.people.knownModel.sortOrder : "Name A–Z"
                options: ["Name A–Z", "Most photos", "Most faces"]
                onOptionSelected: function(value) { if (root.people) root.people.knownModel.setSort(value) }
                Accessible.name: "Sort known people: " + label
            }
            Item { Layout.fillWidth: true }
        }
        Text {
            objectName: "knownPeopleStateText"
            Layout.fillWidth: true
            visible: root.people === null || root.people.state !== "COMPLETE" || root.people.knownModel.visibleCount === 0
            text: !root.people || root.people.state !== "COMPLETE"
                ? (root.people && root.people.state === "FAILED"
                    ? "People analysis is unavailable. " + root.people.stageError
                    : "Processing people…")
                : root.people.knownCount === 0 ? "No known people were found in this project."
                : "No names match your search."
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            font.pixelSize: typography.body
        }
        Components.PillButton {
            objectName: "knownPeopleRetry"
            variant: "accent"
            visible: root.people !== null && root.people.state === "FAILED"
            text: "Retry people analysis"
            onClicked: if (root.people) root.people.retryProcessing()
            Accessible.name: "Retry project processing"
        }
        Components.PeopleGrid {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.people !== null && root.people.state === "COMPLETE"
            peopleModel: root.people ? root.people.knownModel : null
            viewSize: root.viewSize
            known: true
            onPersonOpened: function(key) { if (root.people) root.people.openKnown(key) }
        }
    }
    Components.PeopleDetailPhotos {
        anchors.fill: parent
        visible: root.detailOpen
        person: root.people ? root.people.selectedKnown : ({})
        title: person.name || "Known person"
        photoModel: root.people ? root.people.detailPhotoModel : null
        viewSize: root.viewSize
        inspector: root.inspector
        onBackRequested: if (root.people) root.people.back()
    }
}
